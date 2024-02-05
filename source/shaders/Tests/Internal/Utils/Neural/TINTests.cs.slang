/***************************************************************************
 # Copyright (c) 2015-23, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Utils.Neural.TIN.TinMLP;
import Internal.Utils.Neural.TIN.Tin;
import Internal.Utils.Neural.ActivationFunctions;

RWStructuredBuffer<float> result;

StructuredBuffer<Tin::MMAMatStore> tinWeights;
StructuredBuffer<uint> tinBiases;

[numthreads(32, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    Tin::HPackedArray<16> ip_arr;

    [unroll]
    for (uint i = 0; i < 16; i++)
    {
        ip_arr[i] = i + 1;
    }

    /*** start tin operations **/
    Tin::HVector<16> ip;
    ip.clear();
    ip.from_array(ip_arr);

    Tin::HMLP<1, 16, 32, 16, Act::HReLU, Act::HNone> mlp;
    let res = mlp.forward(ip, tinWeights, tinBiases);

    Tin::HPackedArray<16> op_arr;
    res.to_array(op_arr);
    /*** end tin operations **/

    if (WaveIsFirstLane())
    {
        [unroll]
        for (uint i = 0; i < 16; i++)
        {
            result[i] = op_arr[i];
        }
    }
}
