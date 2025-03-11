//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#pragma once

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

	MipGenerator();
	virtual ~MipGenerator();

	bool Init(XUSG::CommandList* pCommandList, const XUSG::DescriptorTableLib::sptr& descriptorTableLib,
		std::vector<XUSG::Resource::uptr>& uploaders, XUSG::Format rtFormat, const char* fileName, bool typedUAV);

	void Process(XUSG::CommandList* pCommandList, XUSG::ResourceState dstState, PipelineType pipelineType);
	void Visualize(XUSG::CommandList* pCommandList, XUSG::RenderTarget* pRenderTarget, uint32_t mipLevel);

	uint32_t GetMipLevelCount() const;
	void GetImageSize(uint32_t& width, uint32_t& height) const;

protected:
	enum PipelineIndex : uint8_t
	{
		BLIT_2D_GRAPHICS,
		BLIT_2D_COMPUTE,
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

	XUSG::ShaderLib::uptr				m_shaderLib;
	XUSG::Graphics::PipelineLib::uptr	m_graphicsPipelineLib;
	XUSG::Compute::PipelineLib::uptr	m_computePipelineLib;
	XUSG::PipelineLayoutLib::uptr		m_pipelineLayoutLib;
	XUSG::DescriptorTableLib::sptr		m_descriptorTableLib;

	XUSG::PipelineLayout	m_pipelineLayouts[NUM_PIPELINE];
	XUSG::Pipeline			m_pipelines[NUM_PIPELINE];

	std::vector<XUSG::DescriptorTable>	m_uavTables[NUM_UAV_TABLE_TYPE];
	std::vector<XUSG::DescriptorTable>	m_srvTables;
	XUSG::DescriptorTable				m_uavTable;
	XUSG::DescriptorTable				m_srvTable;
	XUSG::DescriptorTable				m_samplerTable;

	XUSG::Texture::uptr					m_source;
	XUSG::TypedBuffer::uptr				m_counter;
	XUSG::RenderTarget::uptr			m_mipmaps;

	DirectX::XMUINT2					m_imageSize;

	XUSG::ResourceBarrier				m_barriers[2];
	uint32_t							m_numBarriers;

	bool								m_typedUAV;
};
