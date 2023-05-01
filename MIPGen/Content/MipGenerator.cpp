//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#include "MipGenerator.h"
#define _INDEPENDENT_DDS_LOADER_
#include "Advanced/XUSGDDSLoader.h"
#undef _INDEPENDENT_DDS_LOADER_

using namespace std;
using namespace DirectX;
using namespace XUSG;

MipGenerator::MipGenerator() :
	m_imageSize(1, 1)
{
	m_shaderLib = ShaderLib::MakeUnique();
}

MipGenerator::~MipGenerator()
{
}

bool MipGenerator::Init(CommandList* pCommandList, const DescriptorTableLib::sptr& descriptorTableLib,
	vector<Resource::uptr>& uploaders, Format rtFormat, const wchar_t* fileName, bool typedUAV)
{
	const auto pDevice = pCommandList->GetDevice();
	m_graphicsPipelineLib = Graphics::PipelineLib::MakeUnique(pDevice);
	m_computePipelineLib = Compute::PipelineLib::MakeUnique(pDevice);
	m_pipelineLayoutLib = PipelineLayoutLib::MakeUnique(pDevice);
	m_descriptorTableLib = descriptorTableLib;

	m_typedUAV = typedUAV;

	// Load input image
	{
		DDS::Loader textureLoader;
		DDS::AlphaMode alphaMode;

		uploaders.emplace_back(Resource::MakeUnique());
		XUSG_N_RETURN(textureLoader.CreateTextureFromFile(pCommandList, fileName,
			8192, false, m_source, uploaders.back().get(), &alphaMode), false);
	}

	// Create resources and pipelines
	m_imageSize.x = static_cast<uint32_t>(m_source->GetWidth());
	m_imageSize.y = m_source->GetHeight();
	const uint8_t numMips = CalculateMipLevels(m_imageSize.x, m_imageSize.y);

	m_counter = TypedBuffer::MakeUnique();
	XUSG_N_RETURN(m_counter->Create(pDevice, 1, sizeof(uint32_t), Format::R32_UINT,
		ResourceFlag::ALLOW_UNORDERED_ACCESS | ResourceFlag::DENY_SHADER_RESOURCE,
		MemoryType::DEFAULT, 0, nullptr, 1, nullptr, MemoryFlag::NONE,
		L"GlobalBarrierCounter"), false);

	m_mipmaps = RenderTarget::MakeUnique();
	XUSG_N_RETURN(m_mipmaps->Create(pDevice, m_imageSize.x, m_imageSize.y, rtFormat, 1, (typedUAV ?
		ResourceFlag::ALLOW_UNORDERED_ACCESS : ResourceFlag::NEED_PACKED_UAV ) |
		ResourceFlag::ALLOW_SIMULTANEOUS_ACCESS, numMips, 1, nullptr, false,
		MemoryFlag::NONE, L"MipMap"), false);

	XUSG_N_RETURN(createPipelineLayouts(), false);
	XUSG_N_RETURN(createPipelines(rtFormat), false);
	XUSG_N_RETURN(createDescriptorTables(), false);

	{
		// Set the descriptor heaps
		const DescriptorHeap descriptorHeaps[] =
		{
			m_descriptorTableLib->GetDescriptorHeap(CBV_SRV_UAV_HEAP),
			m_descriptorTableLib->GetDescriptorHeap(SAMPLER_HEAP)
		};
		pCommandList->SetDescriptorHeaps(static_cast<uint32_t>(size(descriptorHeaps)), descriptorHeaps);

		// Auto promotion to UNORDERED_ACCESS
		m_mipmaps->SetBarrier(m_barriers, ResourceState::UNORDERED_ACCESS);

		pCommandList->SetComputePipelineLayout(m_pipelineLayouts[RESAMPLE_COMPUTE]);
		m_mipmaps->AsTexture()->Blit(pCommandList, 8, 8, 1, m_uavTables[UAV_TABLE_TYPED][0], 1,
			0, m_srvTable, 2, m_samplerTable, 0, m_pipelines[RESAMPLE_COMPUTE]);
	}

	return true;
}

void MipGenerator::Process(CommandList* pCommandList, ResourceState dstState, PipelineType pipelineType)
{
	switch (pipelineType)
	{
	case COMPUTE:
		m_numBarriers = generateMipsCompute(pCommandList, m_barriers, dstState);
		break;
	case SINGLE_PASS:
		m_numBarriers = generateMipsSinglePass(pCommandList, m_barriers, dstState);
		break;
	default:
		m_numBarriers = generateMipsGraphics(pCommandList, m_barriers, dstState);
	}
}

void MipGenerator::Visualize(CommandList* pCommandList, RenderTarget* pRenderTarget, uint32_t mipLevel)
{
	m_numBarriers = pRenderTarget->SetBarrier(m_barriers, ResourceState::RENDER_TARGET, m_numBarriers);
	pCommandList->Barrier(m_numBarriers, m_barriers);

	pCommandList->SetGraphicsPipelineLayout(m_pipelineLayouts[RESAMPLE_GRAPHICS]);
	pRenderTarget->Blit(pCommandList, m_srvTables[mipLevel], 1, 0, 0, 0,
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
		XUSG_X_RETURN(m_pipelineLayouts[RESAMPLE_GRAPHICS], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutLib.get(), PipelineLayoutFlag::NONE, L"ResamplingGraphicsLayout"), false);
	}

	// Resampling compute
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetRange(0, DescriptorType::SAMPLER, 1, 0);
		utilPipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		utilPipelineLayout->SetRange(2, DescriptorType::SRV, 1, 0);
		XUSG_X_RETURN(m_pipelineLayouts[RESAMPLE_COMPUTE], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutLib.get(), PipelineLayoutFlag::NONE, L"ResamplingComputeLayout"), false);
	}

	// One-pass MIP-Gen
	{
		const auto utilPipelineLayout = Util::PipelineLayout::MakeUnique();
		utilPipelineLayout->SetConstants(0, XUSG_UINT32_SIZE_OF(uint32_t[2]), 0);
		utilPipelineLayout->SetRange(1, DescriptorType::UAV, 1, 0, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		utilPipelineLayout->SetRange(2, DescriptorType::UAV, m_mipmaps->GetNumMips(), 1, 0, DescriptorFlag::DATA_STATIC_WHILE_SET_AT_EXECUTE);
		XUSG_X_RETURN(m_pipelineLayouts[SINGLE_PASS_MIPGEN], utilPipelineLayout->GetPipelineLayout(
			m_pipelineLayoutLib.get(), PipelineLayoutFlag::NONE, L"OnePassMIPGenLayout"), false);
	}

	return true;
}

bool MipGenerator::createPipelines(Format rtFormat)
{
	auto vsIndex = 0u;
	auto psIndex = 0u;
	auto csIndex = 0u;

	// Resampling graphics
	XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::VS, vsIndex, L"VSScreenQuad.cso"), false);
	{
		XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::PS, psIndex, L"PSResample.cso"), false);

		const auto state = Graphics::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[RESAMPLE_GRAPHICS]);
		state->SetShader(Shader::Stage::VS, m_shaderLib->GetShader(Shader::Stage::VS, vsIndex));
		state->SetShader(Shader::Stage::PS, m_shaderLib->GetShader(Shader::Stage::PS, psIndex++));
		state->DSSetState(Graphics::DEPTH_STENCIL_NONE, m_graphicsPipelineLib.get());
		state->IASetPrimitiveTopologyType(PrimitiveTopologyType::TRIANGLE);
		state->OMSetNumRenderTargets(1);
		state->OMSetRTVFormat(0, rtFormat);
		XUSG_X_RETURN(m_pipelines[RESAMPLE_GRAPHICS], state->GetPipeline(m_graphicsPipelineLib.get(), L"Resampling_graphics"), false);
	}

	// Resampling compute
	{
		XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, csIndex, L"CSResample.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[RESAMPLE_COMPUTE]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, csIndex++));
		XUSG_X_RETURN(m_pipelines[RESAMPLE_COMPUTE], state->GetPipeline(m_computePipelineLib.get(), L"Resampling_compute"), false);
	}

	// One-pass MIP-Gen
	{
		XUSG_N_RETURN(m_shaderLib->CreateShader(Shader::Stage::CS, csIndex, m_typedUAV ? L"CSGenerateMips.cso" : L"CSGenMipsPacked.cso"), false);

		const auto state = Compute::State::MakeUnique();
		state->SetPipelineLayout(m_pipelineLayouts[SINGLE_PASS_MIPGEN]);
		state->SetShader(m_shaderLib->GetShader(Shader::Stage::CS, csIndex++));
		XUSG_X_RETURN(m_pipelines[SINGLE_PASS_MIPGEN], state->GetPipeline(m_computePipelineLib.get(), L"OnePassMIPGen"), false);
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
		XUSG_X_RETURN(m_uavTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	// Get UAVs for resampling
	m_uavTables[UAV_TABLE_TYPED].resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		// Get UAV
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetUAV(i));
		XUSG_X_RETURN(m_uavTables[UAV_TABLE_TYPED][i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	if (!m_typedUAV)
	{
		m_uavTables[UAV_TABLE_PACKED].resize(numMips);
		for (uint8_t i = 0; i < numMips; ++i)
		{
			// Get UAV
			const auto descriptorTable = Util::DescriptorTable::MakeUnique();
			descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetPackedUAV(i));
			XUSG_X_RETURN(m_uavTables[UAV_TABLE_PACKED][i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
		}
	}

	// Get SRVs for resampling
	m_srvTables.resize(numMips);
	for (uint8_t i = 0; i < numMips; ++i)
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_mipmaps->GetSRVLevel(i));
		XUSG_X_RETURN(m_srvTables[i], descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	// Get SRV for source blit
	{
		const auto descriptorTable = Util::DescriptorTable::MakeUnique();
		descriptorTable->SetDescriptors(0, 1, &m_source->GetSRV());
		XUSG_X_RETURN(m_srvTable, descriptorTable->GetCbvSrvUavTable(m_descriptorTableLib.get()), false);
	}

	// Create the sampler table
	const auto descriptorTable = Util::DescriptorTable::MakeUnique();
	const auto sampler = LINEAR_CLAMP;
	descriptorTable->SetSamplers(0, 1, &sampler, m_descriptorTableLib.get());
	XUSG_X_RETURN(m_samplerTable, descriptorTable->GetSamplerTable(m_descriptorTableLib.get()), false);

	return true;
}

uint32_t MipGenerator::generateMipsGraphics(CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	// Generate mipmaps
	return m_mipmaps->GenerateMips(pCommandList, pBarriers, ResourceState::PIXEL_SHADER_RESOURCE |
		ResourceState::NON_PIXEL_SHADER_RESOURCE, m_pipelineLayouts[RESAMPLE_GRAPHICS],
		m_pipelines[RESAMPLE_GRAPHICS], m_srvTables.data(), 1, m_samplerTable, 0);
}

uint32_t MipGenerator::generateMipsCompute(CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	// Generate mipmaps
	return m_mipmaps->AsTexture()->GenerateMips(pCommandList, pBarriers, 8, 8, 1,
		ResourceState::PIXEL_SHADER_RESOURCE | ResourceState::NON_PIXEL_SHADER_RESOURCE,
		m_pipelineLayouts[RESAMPLE_COMPUTE], m_pipelines[RESAMPLE_COMPUTE],
		&m_uavTables[UAV_TABLE_TYPED][1], 1, m_samplerTable, 0, 0, &m_srvTables[0], 2);
}

uint32_t MipGenerator::generateMipsSinglePass(CommandList* pCommandList,
	ResourceBarrier* pBarriers, ResourceState dstState)
{
	const auto groupCountX = XUSG_DIV_UP(static_cast<uint32_t>(m_mipmaps->GetWidth()), 32);
	const auto groupCountY = XUSG_DIV_UP(m_mipmaps->GetHeight(), 32);

	// Clear counter
	const uint32_t clear[4] = {};
	pCommandList->ClearUnorderedAccessViewUint(m_uavTable, m_counter->GetUAV(), m_counter.get(), clear);

	pCommandList->SetComputePipelineLayout(m_pipelineLayouts[SINGLE_PASS_MIPGEN]);
	pCommandList->SetCompute32BitConstant(0, m_mipmaps->GetNumMips());
	pCommandList->SetCompute32BitConstant(0, groupCountX * groupCountY, XUSG_UINT32_SIZE_OF(uint32_t));
	pCommandList->SetComputeDescriptorTable(1, m_uavTable);
	pCommandList->SetComputeDescriptorTable(2, m_uavTables[m_typedUAV ? UAV_TABLE_TYPED : UAV_TABLE_PACKED][0]);

	pCommandList->SetPipelineState(m_pipelines[SINGLE_PASS_MIPGEN]);

	// Auto promotion to UNORDERED_ACCESS
	m_mipmaps->SetBarrier(m_barriers, ResourceState::UNORDERED_ACCESS);

	pCommandList->Dispatch(groupCountX, groupCountY, 1);

	return m_mipmaps->SetBarrier(m_barriers, dstState);
}
