//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#define WAVE_SIZE_X 8

#ifndef __D3DX_DXGI_FORMAT_CONVERT_INL___
#define D3DX_R8G8B8A8_UNORM_to_FLOAT4(x) (x)
#define D3DX_FLOAT4_to_R8G8B8A8_UNORM(x) (x)
typedef float4 T;
#else
typedef uint T;
#endif

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cb
{
	uint g_numMips;
	uint g_numGroups;
};

static const uint2 g_offsets2x2[] = { uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(1, 1) };

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
globallycoherent RWBuffer<uint> g_rwCounter;
RWTexture2D<T> g_txMipMaps[] : register (u1);

groupshared float4 g_groupVals[32][32];
groupshared uint g_counter;

float4 sample2x2(RWTexture2D<T> tex, uint2 pos)
{
	const uint2 pos00 = pos * 2;
	float4 sum = 0.0;

	[unroll]
	for (uint i = 0; i < 4; ++i)
		sum += D3DX_R8G8B8A8_UNORM_to_FLOAT4(tex[pos00 + g_offsets2x2[i]]);

	return sum / 4.0;
}

#ifndef HLSL_VERSION
//--------------------------------------------------------------------------------------
// Per 4x4 tile process within a wave
//--------------------------------------------------------------------------------------
uint Per4x4TileProcess(float4 val, uint level, uint2 dTid, uint2 gTid)
{
	static const uint tileSize = 4;
	
	const uint2 tTid = dTid % tileSize;	// Tile thread Id
	const uint2 tid = dTid / tileSize;	// Tile Id

	const uint2 waveXY = uint2(WaveGetLaneIndex() % WAVE_SIZE_X, WaveGetLaneIndex() / WAVE_SIZE_X);
	const uint2 parity = abs(waveXY / tileSize);
	const uint2 lane00 = parity * tileSize + (1 << tTid);

	uint fillSize = tileSize;

	// For a tile, 4x4 => 1x1
	[unroll]
	for (uint i = 0; i < 2; ++i)
	{
		if (++level >= g_numMips) return 0xffffffff;
		fillSize >>= 1;

		float4 sum = 0.0;

		[unroll]
		for (uint j = 0; j < 4; ++j)
		{
			const uint2 lanePos = lane00 + g_offsets2x2[j];
			uint lane = WAVE_SIZE_X * lanePos.y + lanePos.x;
			lane = min(lane, WaveGetLaneCount() - 1);
			sum += WaveReadLaneAt(val, lane);
		}

		val = sum / 4.0;

		if (all(tTid < fillSize))
			g_txMipMaps[level][tid * fillSize + tTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);
	}

	if (all(gTid % tileSize) == 0)
		g_groupVals[gTid.y / tileSize][gTid.x / tileSize] = val;
	GroupMemoryBarrierWithGroupSync();

	return level;
}

//--------------------------------------------------------------------------------------
// Per group process
//--------------------------------------------------------------------------------------
uint PerGroupProcessW(float4 val, uint level, uint2 dTid, uint2 gTid)
{
	// For a group, 32x32 => 8x8
	level = Per4x4TileProcess(val, level, dTid, gTid);
	if (level == 0xffffffff) return level;

	// For a group, 8x8 => 2x2
	if (all(gTid <= 8))
	{
		// ...
	}

	// For a group, 8x8 => 1x1

	return level;
}
#endif

//--------------------------------------------------------------------------------------
// Per group process
//--------------------------------------------------------------------------------------
uint PerGroupProcess(float4 val, uint level, uint2 gTid, uint2 gid)
{
	g_groupVals[gTid.y][gTid.x] = val;
	uint fillSize = 32;

	// For a group, 32x32 => 1x1
	[unroll]
	for (uint i = 0; i < 5; ++i)
	{
		if (++level >= g_numMips) return 0xffffffff;
		fillSize >>= 1;

		GroupMemoryBarrierWithGroupSync();

		const bool active = all(gTid < fillSize);
		if (active)
		{
			float4 sum = 0.0;

			[unroll]
			for (uint j = 0; j < 4; ++j)
			{
				const uint2 idx = gTid * 2 + g_offsets2x2[j];
				sum += g_groupVals[idx.y][idx.x];
			}

			val = sum / 4.0;

			g_txMipMaps[level][gid * fillSize + gTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);
		}

		GroupMemoryBarrierWithGroupSync();

		if (active) g_groupVals[gTid.y][gTid.x] = val;
	}

	return level;
}

//--------------------------------------------------------------------------------------
// If the current group is the slowest
//--------------------------------------------------------------------------------------
bool IsSlowestGroup(uint gIdx : SV_GroupIndex)
{
	if (gIdx == 0) InterlockedAdd(g_rwCounter[0], 1, g_counter);
	AllMemoryBarrierWithGroupSync();

	return g_counter + 1 == g_numGroups;
}

//--------------------------------------------------------------------------------------
// One-pass MIP-map generation for max texture size of 4096x4096
//--------------------------------------------------------------------------------------
[numthreads(32, 32, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 GTid : SV_GroupThreadID,
	uint2 Gid : SV_GroupID, uint GIdx : SV_GroupIndex)
{
	uint level = 0;
	float4 val;

	if (level + 1 < g_numMips)
	{
		val = sample2x2(g_txMipMaps[level++], DTid);
		g_txMipMaps[level][DTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);

		// For each group, 32x32 => 1x1
		level = PerGroupProcess(val, level, GTid, Gid);
	}

	if (level == 0xffffffff) return;
	if (!IsSlowestGroup(GIdx)) return;

	if (level + 1 < g_numMips)
	{
		val = sample2x2(g_txMipMaps[level++], GTid);
		g_txMipMaps[level][GTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);

		// For the slowest group, 32x32 => 1x1
		PerGroupProcess(val, level, GTid, 0);
	}
}
