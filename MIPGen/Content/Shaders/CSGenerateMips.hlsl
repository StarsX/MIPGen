//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

#ifndef __D3DX_DXGI_FORMAT_CONVERT_INL___
#define D3DX_R8G8B8A8_UNORM_to_FLOAT4(x) (x)
#define D3DX_FLOAT4_to_R8G8B8A8_UNORM(x) (x)
typedef float4 T;
#else
typedef uint T;
#endif

#define GROUP_SIZE	32
#define TILE_SIZE	4

//--------------------------------------------------------------------------------------
// Constant buffer
//--------------------------------------------------------------------------------------
cbuffer cb
{
	uint g_numMips;
	uint g_numGroups;
};

static const uint2 g_offsets2x2[] = { uint2(0, 0), uint2(1, 0), uint2(0, 1), uint2(1, 1) };
static const uint g_tileSize_sq = TILE_SIZE * TILE_SIZE;

//--------------------------------------------------------------------------------------
// Textures
//--------------------------------------------------------------------------------------
globallycoherent RWBuffer<uint> g_rwCounter;
RWTexture2D<T> g_txMipMaps[] : register (u1);

#ifdef HLSL_VERSION
groupshared float4 g_groupVals[GROUP_SIZE][GROUP_SIZE];
#else
groupshared float4 g_groupVals[8][8];
#endif
groupshared uint g_counter;

//--------------------------------------------------------------------------------------
// Reverse Morton code
//--------------------------------------------------------------------------------------
uint ReverseMorton(uint x)
{
	x &= 0x55555555;
	x = (x ^ (x >> 1)) & 0x33333333;
	x = (x ^ (x >> 2)) & 0x0f0f0f0f;
	x = (x ^ (x >> 4)) & 0x00ff00ff;
	x = (x ^ (x >> 8)) & 0x0000ffff;
	return x;
}

//--------------------------------------------------------------------------------------
// Decodes one 32-bit morton code into two 16-bit integers
//--------------------------------------------------------------------------------------
uint2 MortonDecode(uint idx)
{
	return uint2(ReverseMorton(idx), ReverseMorton(idx >> 1));
}

//--------------------------------------------------------------------------------------
// Down-sample texture
//--------------------------------------------------------------------------------------
float4 DownSample(RWTexture2D<T> tex, uint2 pos)
{
	const uint2 pos00 = pos * 2;
	float4 sum = 0.0;

	[unroll]
	for (uint i = 0; i < 4; ++i)
		sum += D3DX_R8G8B8A8_UNORM_to_FLOAT4(tex[pos00 + g_offsets2x2[i]]);

	return sum / 4.0;
}

#ifdef HLSL_VERSION
//--------------------------------------------------------------------------------------
// Per group process
//--------------------------------------------------------------------------------------
uint PerGroupProcess(float4 val, uint level, uint2 dTid, uint2 gTid, uint2 gid, uint gIdx)
{
	g_groupVals[gTid.y][gTid.x] = val;
	uint fillSize = GROUP_SIZE;

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

			g_txMipMaps[level][fillSize * gid + gTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);
		}

		GroupMemoryBarrierWithGroupSync();

		if (active) g_groupVals[gTid.y][gTid.x] = val;
	}

	return level;
}

#else
//--------------------------------------------------------------------------------------
// Per 4x4 tile process within a wave
//--------------------------------------------------------------------------------------
uint PerTileProcess(inout float4 val, uint level, uint2 tTid, uint2 tid)
{
	const uint lane = WaveGetLaneIndex();
	const uint laneBase = g_tileSize_sq * (lane / g_tileSize_sq) + TILE_SIZE * (lane % g_tileSize_sq);

	uint fillSize = TILE_SIZE;

	// For a tile, 4x4 => 1x1
	[unroll]
	for (uint k = 0; k < 2; ++k)
	{
		if (++level >= g_numMips) return 0xffffffff;
		fillSize >>= 1;

		float4 sum = 0.0;

		[unroll]
		for (uint i = 0; i < 4; ++i)
			sum += WaveReadLaneAt(val, laneBase + i);

		val = sum / 4.0;

		if (all(tTid < fillSize))
			g_txMipMaps[level][tid * fillSize + tTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);
	}

	return level;
}

//--------------------------------------------------------------------------------------
// Per group process
//--------------------------------------------------------------------------------------
uint PerGroupProcess(float4 val, uint level, uint2 dTid, uint2 gTid, uint2 gid, uint gIdx)
{
	// For a group, 32x32 => 8x8
	const uint2 tTid = dTid % TILE_SIZE;	// Tile thread Id
	const uint2 tid = dTid / TILE_SIZE;		// Tile Id
	level = PerTileProcess(val, level, tTid, tid);

	// For a group, 8x8 => 4x4
	if (++level >= g_numMips) return 0xffffffff;

	if (gIdx % g_tileSize_sq == 0)
		g_groupVals[gTid.y / TILE_SIZE][gTid.x / TILE_SIZE] = val;
	GroupMemoryBarrierWithGroupSync();

	if (gIdx < g_tileSize_sq)
	{
		const uint2 idx00 = gTid * 2;
		float4 sum = 0.0;

		[unroll]
		for (uint i = 0; i < 4; ++i)
		{
			const uint2 idx = idx00 + g_offsets2x2[i];
			sum += g_groupVals[idx.y][idx.x];
		}

		val = sum / 4.0;

		g_txMipMaps[level][TILE_SIZE * gid + gTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);
	}

	// For a group, 4x4 => 1x1
	return PerTileProcess(val, level, gTid, gid);
}
#endif

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
[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint2 DTid : SV_DispatchThreadID, uint2 GTid : SV_GroupThreadID,
	uint GIdx : SV_GroupIndex, uint2 Gid : SV_GroupID)
{
#if 0
	const uint2 gTid = GTid;
	const uint2 dTid = DTid;
#else
	const uint2 gTid = MortonDecode(GIdx);
	const uint2 dTid = GROUP_SIZE * Gid + gTid;
#endif

	uint level = 0;
	float4 val;

	if (level + 1 < g_numMips)
	{
		val = DownSample(g_txMipMaps[level++], dTid);
		g_txMipMaps[level][dTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);

		// For each group, 32x32 => 1x1
		level = PerGroupProcess(val, level, dTid, gTid, Gid, GIdx);
	}

	if (!IsSlowestGroup(GIdx)) return;

	if (level + 1 < g_numMips)
	{
		val = DownSample(g_txMipMaps[level++], gTid);
		g_txMipMaps[level][gTid] = D3DX_FLOAT4_to_R8G8B8A8_UNORM(val);

		// For the slowest group, 32x32 => 1x1
		PerGroupProcess(val, level, gTid, gTid, 0, GIdx);
	}
}
