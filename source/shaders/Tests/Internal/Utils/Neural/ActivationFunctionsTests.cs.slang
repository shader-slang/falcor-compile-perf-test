/***************************************************************************
 # Copyright (c) 2015-22, NVIDIA CORPORATION. All rights reserved.
 #
 # NVIDIA CORPORATION and its licensors retain all intellectual property
 # and proprietary rights in and to this software, related documentation
 # and any modifications thereto.  Any use, reproduction, disclosure or
 # distribution of this software and related documentation without an express
 # license agreement from NVIDIA CORPORATION is strictly prohibited.
 **************************************************************************/
import Internal.Utils.Neural.ActivationFunctions;

RWStructuredBuffer<float> result;
StructuredBuffer<float> input;

[numthreads(256, 1, 1)]
void main(uint3 dispatchThreadID: SV_DispatchThreadID)
{
    const uint i = dispatchThreadID.x;

    let activation = ACTIVATION_FUNCTION();

    // Evaluate activation function.
    // Note that the input/output will be up/down-cast to the precision used by the function.
    float x = input[i];
    float y = activation.eval(x);

    result[i] = y;
}
