//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

#include "DXFramework.h"
#include "Core/XUSG.h"

class MipGenerator
{
public:
	enum PipelineType
	{
		GRAPHICS,
		COMPUTE,
		SINGLE_PASS,

		NUM_PIPE_TYPE
	};

	MipGenerator(const XUSG::Device::sptr& device);
	virtual ~MipGenerator();

	bool Init(XUSG::CommandList* pCommandList, std::vector<XUSG::Resource::uptr>& uploaders,
		XUSG::Format rtFormat, const wchar_t* fileName, bool typedUAV);

	void Process(XUSG::CommandList* pCommandList, XUSG::ResourceState dstState, PipelineType pipelineType);
	void Visualize(XUSG::CommandList* pCommandList, XUSG::RenderTarget::uptr& renderTarget, uint32_t mipLevel);

	uint32_t GetMipLevelCount() const;
	void GetImageSize(uint32_t& width, uint32_t& height) const;

protected:
	enum PipelineIndex : uint8_t
	{
		RESAMPLE_GRAPHICS,
		RESAMPLE_COMPUTE,
		SINGLE_PASS_MIPGEN,

		NUM_PIPELINE
	};

	enum UavTableType : uint8_t
	{
		UAV_TABLE_TYPED,
		UAV_TABLE_PACKED,

		NUM_UAV_TABLE_TYPE
	};

	bool createPipelineLayouts();
	bool createPipelines(XUSG::Format rtFormat);
	bool createDescriptorTables();

	uint32_t generateMipsGraphics(XUSG::CommandList* pCommandList,
		XUSG::ResourceBarrier* pBarriers, XUSG::ResourceState dstState);
	uint32_t generateMipsCompute(XUSG::CommandList* pCommandList,
		XUSG::ResourceBarrier* pBarriers , XUSG::ResourceState dstState);
	uint32_t generateMipsSinglePass(XUSG::CommandList* pCommandList,
		XUSG::ResourceBarrier* pBarriers, XUSG::ResourceState dstState);

	XUSG::Device::sptr m_device;

	XUSG::ShaderPool::uptr				m_shaderPool;
	XUSG::Graphics::PipelineCache::uptr	m_graphicsPipelineCache;
	XUSG::Compute::PipelineCache::uptr	m_computePipelineCache;
	XUSG::PipelineLayoutCache::uptr		m_pipelineLayoutCache;
	XUSG::DescriptorTableCache::uptr	m_descriptorTableCache;

	XUSG::PipelineLayout	m_pipelineLayouts[NUM_PIPELINE];
	XUSG::Pipeline			m_pipelines[NUM_PIPELINE];

	std::vector<XUSG::DescriptorTable>	m_uavTables[NUM_UAV_TABLE_TYPE];
	std::vector<XUSG::DescriptorTable>	m_srvTables;
	XUSG::DescriptorTable				m_uavTable;
	XUSG::DescriptorTable				m_srvTable;
	XUSG::DescriptorTable				m_samplerTable;

	XUSG::Texture::sptr					m_source;
	XUSG::TypedBuffer::uptr				m_counter;
	XUSG::RenderTarget::uptr			m_mipmaps;

	DirectX::XMUINT2					m_imageSize;

	XUSG::ResourceBarrier				m_barriers[2];
	uint32_t							m_numBarriers;

	bool								m_typedUAV;
};
