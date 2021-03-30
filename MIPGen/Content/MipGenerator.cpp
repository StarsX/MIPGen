//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "MipGenerator.h"
#define _INDEPENDENT_DDS_LOADER_
#include "Advanced/XUSGDDSLoader.h"
#undef _INDEPENDENT_DDS_LOADER_

#define SizeOfInUint32(obj)	DIV_UP(sizeof(obj), sizeof(uint32_t))

using namespace std;
using namespace DirectX;
using namespace XUSG;

struct GaussianConstants
{
	struct Immutable
	{
		XMFLOAT2	Focus;
		float		Sigma;
	} Imm;
	uint32_t Level;
};

MipGenerator::MipGenerator(const Device& device) :
	m_device(device),
	m_imageSize(1, 1)
{
	m_shaderPool = ShaderPool::MakeUnique();
	m_graphicsPipelineCache = Graphics::PipelineCache::MakeUnique(device);
	m_computePipelineCache = Compute::PipelineCache::MakeUnique(device);
	m_descriptorTableCache = DescriptorTableCache::MakeUnique(device);
	m_pipelineLayoutCache = PipelineLayoutCache::MakeUnique(device);
}

MipGenerator::~MipGenerator()
{
}

bool MipGenerator::Init(CommandList* pCommandList,  vector<Resource>& uploaders,
	Format rtFormat, const wchar_t* fileName, bool typedUAV)
{
	m_typedUAV = typedUAV;

	// Load input image
	{
		DDS::Loader textureLoader;
		DDS::AlphaMode alphaMode;

		uploaders.push_back(nullptr);
		N_RETURN(textureLoader.CreateTextureFromFile(m_device, pCommandList, fileName,
			8192, false, m_source, uploaders.back(), &alphaMode), false);
	}

	// Create resources and pipelines
	m_imageSize.x = static_cast<uint32_t>(m_source->GetResource()->GetDesc().Width);
	m_imageSize.y = m_source->GetResource()->GetDesc().Height;
	const uint8_t numMips = max<uint8_t>(Log2((max)(m_imageSize.x, m_imageSize.y)), 0) + 1;

	m_counter = TypedBuffer::MakeUnique();
	N_RETURN(m_counter->Create(m_device, 1, sizeof(uint32_t), Format::R32_FLOAT,
		ResourceFlag::ALLOW_UNORDERED_ACCESS | ResourceFlag::DENY_SHADER_RESOURCE,
		MemoryType::DEFAULT, 0, nullptr, 1, nullptr, L"GlobalBarrierCounter"), false);

	m_mipmaps = RenderTarget::MakeUnique();
	N_RETURN(m_mipmaps->Create(m_device, m_imageSize.x, m_imageSize.y, rtFormat, 1, (typedUAV ?
		ResourceFlag::ALLOW_UNORDERED_ACCESS : ResourceFlag::NEED_PACKED_UAV ) |
		ResourceFlag::ALLOW_SIMULTANEOUS_ACCESS, numMips, 1, nullptr, false, L"MipMap"), false);

	N_RETURN(createPipelineLayouts(), false);
	N_RETURN(createPipelines(rtFormat), false);
	N_RETURN(createDescriptorTables(), false);

	{
		// Set Descriptor pools
		const DescriptorPool descriptorPools[] =
		{
			m_descriptorTableCache->GetDescriptorPool(CBV_SRV_UAV_POOL),
			m_descriptorTableCache->GetDescriptorPool(SAMPLER_POOL)
		};
		pCommandList->SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

		// Auto promotion to UNORDERED_ACCESS
		m_mipmaps->SetBarrier(m_barriers, ResourceState::UNORDERED_ACCESS);

		pCommandList->SetComputePipelineLayout(m_pipelineLayouts[RESAMPLE_COMPUTE]);
		m_mipmaps->AsTexture2D()->Blit(pCommandList, 8, 8, 1, m_uavTables[UAV_TABLE_TYPED][0], 1,
			0, m_srvTable, 2, m_samplerTable, 0, m_pipelines[RESAMPLE_COMPUTE]);
	}

	return true;
}

void MipGenerator::Process(const CommandList* pCommandList, ResourceState dstState, PipelineType pipelineType)
{
	// Set Descriptor pools
	const DescriptorPool descriptorPools[] =
	{
		m_descriptorTableCache->GetDescriptorPool(CBV_SRV_UAV_POOL),
		m_descriptorTableCache->GetDescriptorPool(SAMPLER_POOL)
	};
	pCommandList->SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	switch (pipelineType)
	{
	case COMPUTE:
		m_numBarriers = generateMipsCompute(pCommandList, m_barriers, dstState);
		break;
	case ONE_PASS:
		m_numBarriers = generateMipsOnePass(pCommandList, m_barriers, dstState);
		break;
	default:
		m_numBarriers = generateMipsGraphics(pCommandList, m_barriers, dstState);
	}
}

void MipGenerator::Visualize(const CommandList* pCommandList, RenderTarget::uptr& renderTarget, uint32_t mipLevel)
{
	// Set Descriptor pools
	const DescriptorPool descriptorPools[] =
	{
		m_descriptorTableCache->GetDescriptorPool(CBV_SRV_UAV_POOL),
		m_descriptorTableCache->GetDescriptorPool(SAMPLER_POOL)
	};
	pCommandList->SetDescriptorPools(static_cast<uint32_t>(size(descriptorPools)), descriptorPools);

	m_numBarriers = renderTarget->SetBarrier(m_barriers, ResourceState::RENDER_TARGET, m_numBarriers);
	pCommandList->Barrier(m_numBarriers, m_barriers);

	pCommandList->SetGraphicsPipelineLayout(m_pipelineLayouts[RESAMPLE_GRAPHICS]);
	renderTarget->Blit(pCommandList, m_srvTables[mipLevel], 1, 0, 0, 0,
		m_samplerTable, 0, m_pipelines[RESAMPLE_GRAPHICS]);
}

uint32_t MipGenerator::GetMipLevelCount() const
{
	return m_mipmaps->GetNumMips();
}

void MipGenerator::GetImageSize(uint32_t& width, uint32_t& height) const
{
	width = m_imageSize.x;
	height = m_imageSize.y;
}

bool MipGenerator::createPipelineLayouts()
{
	// Resampling graphics
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::SRV, 1, 0);
		utilPipelineLayout->SetShaderStage(0, Shader::PS);
		utilPipelineLayout->SetShaderStage(1, Shader::PS);
		X_RETURN(m_pipelineLayouts[RESAMPLE_GRAPHICS], utilPipelineLayout->GetPipelineLayout(
			*m_pipelineLayoutCache, PipelineLayoutFlag::NONE, L"ResamplingGraphicsLayout"), false);
	}

	// Resampling compute
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		utilPipelineLayout->SetRange(2, DescriptorType::SRV, 1, 0);
		X_RETURN(m_pipelineLayouts[RESAMPLE_COMPUTE], utilPipelineLayout->GetPipelineLayout(
			*m_pipelineLayoutCache, PipelineLayoutFlag::NONE, L"ResamplingComputeLayout"), false);
	}

	// One-pass MIP-Gen
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetConstants(0, SizeOfInUint32(uint32_t[2]), 0);
		utilPipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		utilPipelineLayout->SetRange(2, DescriptorType::UAV, m_mipmaps->GetNumMips(), 1, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		X_RETURN(m_pipelineLayouts[ONE_PASS_MIPGEN], utilPipelineLayout->GetPipelineLayout(
			*m_pipelineLayoutCache, PipelineLayoutFlag::NONE, L"OnePassMIPGenLayout"), false);
	}

	return true;
}

bool MipGenerator::createPipelines(Format rtFormat)
{
	auto vsIndex = 0u;
	auto psIndex = 0u;
	auto csIndex = 0u;

	// Resampling graphics
	N_RETURN(m_shaderPool->CreateShader(Shader::Stage::VS, vsIndex, L"VSScreenQuad.cso"), false);
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::PS, psIndex, L"PSResample.cso"), false);

		const auto state = Graphics::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[RESAMPLE_GRAPHICS]);
		state->SetShader(Shader::Stage::VS, m_shaderPool->GetShader(Shader::Stage::VS, vsIndex));
		state->SetShader(Shader::Stage::PS, m_shaderPool->GetShader(Shader::Stage::PS, psIndex++));
		state->DSSetState(Graphics::DEPTH_STENCIL_NONE, *m_graphicsPipelineCache);
		state->IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
		state->OMSetNumRenderTargets(1);
		state->OMSetRTVFormat(0, rtFormat);
		X_RETURN(m_pipelines[RESAMPLE_GRAPHICS], state->GetPipeline(*m_graphicsPipelineCache, L"Resampling_graphics"), false);
	}

	// Resampling compute
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, L"CSResample.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[RESAMPLE_COMPUTE]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex++));
		X_RETURN(m_pipelines[RESAMPLE_COMPUTE], state->GetPipeline(*m_computePipelineCache, L"Resampling_compute"), false);
	}

	// One-pass MIP-Gen
	{
		N_RETURN(m_shaderPool->CreateShader(Shader::Stage::CS, csIndex, m_typedUAV ? L"CSGenerateMips.cso" : L"CSGenMipsPacked.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[ONE_PASS_MIPGEN]);
		state->SetShader(m_shaderPool->GetShader(Shader::Stage::CS, csIndex++));
		X_RETURN(m_pipelines[ONE_PASS_MIPGEN], state->GetPipeline(*m_computePipelineCache, L"OnePassMIPGen"), false);
	}

	return true;
}

bool MipGenerator::createDescriptorTables()
{
	const auto numMips = m_mipmaps->GetNumMips();

	// Get counter UAV
	{
		// Get UAV
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_counter->GetUAV());
		X_RETURN(m_uavTable, descriptorTable->GetCbvSrvUavTable(*m_descriptorTableCache), false);
	}

	// Get UAVs for resampling
	m_uavTables[UAV_TABLE_TYPED].resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		// Get UAV
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetUAV(i));
		X_RETURN(m_uavTables[UAV_TABLE_TYPED][i], descriptorTable->GetCbvSrvUavTable(*m_descriptorTableCache), false);
	}

	if (!m_typedUAV)
	{
		m_uavTables[UAV_TABLE_PACKED].resize(numMips);
		for (uint8_t i = 0; i < numMips; ++i)
		{
			// Get UAV
			const auto descriptorTable = Util::DescriptorTable::MakeUnique();
			descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetPackedUAV(i));
			X_RETURN(m_uavTables[UAV_TABLE_PACKED][i], descriptorTable->GetCbvSrvUavTable(*m_descriptorTableCache), false);
		}
	}

	// Get SRVs for resampling
	m_srvTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetSRVLevel(i));
		X_RETURN(m_srvTables[i], descriptorTable->GetCbvSrvUavTable(*m_descriptorTableCache), false);
	}

	// Get SRV for source blit
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_source->GetSRV());
		X_RETURN(m_srvTable, descriptorTable->GetCbvSrvUavTable(*m_descriptorTableCache), false);
	}

	// Create the sampler table
	const auto descriptorTable = Util::DescriptorTable::MakeUnique();
	const auto sampler = LINEAR_CLAMP;
	descriptorTable->SetSamplers(0, 1, &sampler, *m_descriptorTableCache);
	X_RETURN(m_samplerTable, descriptorTable->GetSamplerTable(*m_descriptorTableCache), false);

	return true;
}

uint32_t MipGenerator::generateMipsGraphics(const CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	// Generate mipmaps
	return m_mipmaps->GenerateMips(pCommandList, pBarriers, ResourceState::PIXEL_SHADER_RESOURCE |
		ResourceState::NON_PIXEL_SHADER_RESOURCE, m_pipelineLayouts[RESAMPLE_GRAPHICS],
		m_pipelines[RESAMPLE_GRAPHICS], m_srvTables.data(), 1, m_samplerTable, 0);
}

uint32_t MipGenerator::generateMipsCompute(const CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	// Generate mipmaps
	return m_mipmaps->AsTexture2D()->GenerateMips(pCommandList, pBarriers, 8, 8, 1,
		ResourceState::PIXEL_SHADER_RESOURCE | ResourceState::NON_PIXEL_SHADER_RESOURCE,
		m_pipelineLayouts[RESAMPLE_COMPUTE], m_pipelines[RESAMPLE_COMPUTE],
		&m_uavTables[UAV_TABLE_TYPED][1], 1, m_samplerTable, 0, 0, &m_srvTables[0], 2);
}

uint32_t MipGenerator::generateMipsOnePass(const CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	const auto groupCountX = DIV_UP(m_mipmaps->GetWidth(), 32);
	const auto groupCountY = DIV_UP(m_mipmaps->GetHeight(), 32);

	// Clear counter
	const uint32_t clear[4] = {};
	pCommandList->ClearUnorderedAccessViewUint(m_uavTable,
		m_counter->GetUAV(), m_counter->GetResource(), clear);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[ONE_PASS_MIPGEN]);
	pCommandList->SetCompute32BitConstant(0, m_mipmaps->GetNumMips());
	pCommandList->SetCompute32BitConstant(0, groupCountX * groupCountY, SizeOfInUint32(uint32_t));
	pCommandList->SetComputeDescriptorTable(1, m_uavTable);
	pCommandList->SetComputeDescriptorTable(2, m_uavTables[m_typedUAV ? UAV_TABLE_TYPED : UAV_TABLE_PACKED][0]);

	pCommandList->SetPipelineState(m_pipelines[ONE_PASS_MIPGEN]);

	// Auto promotion to UNORDERED_ACCESS
	m_mipmaps->SetBarrier(m_barriers, ResourceState::UNORDERED_ACCESS);

	pCommandList->Dispatch(groupCountX, groupCountY, 1);

	return m_mipmaps->SetBarrier(m_barriers, dstState);
}
