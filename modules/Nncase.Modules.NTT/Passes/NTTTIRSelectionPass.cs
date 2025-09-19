﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes;

public sealed class NTTTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public NTTTIRSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
    }

    protected override Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        var op = call.Target;
        switch (op)
        {
            case IR.Math.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.CustomNTT.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.Math.Clamp clamp:
                return GenerateClamp(call, arguments, output);
            case IR.Distributed.Boxing boxing:
                return GenerateBoxing(call, boxing, arguments, ref output);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, None.Default, binary.BinaryOp, Array.Empty<int>(), Array.Empty<Dimension>(), Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.Tensors.Bitcast bitcast:
                return GenerateBitcast((Expr)arguments[0], ref output, bitcast.NewType);
            case IR.Tensors.Pack pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, pack.Lanes, pack.Axes);
            case IR.Tensors.VectorizeMask pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, new[] { pack.Lanes }, new[] { pack.Axis });
            case IR.Tensors.Unpack unpack:
                return TIR.F.NTT.Unpack((Expr)arguments[0], output, unpack.Lanes, unpack.Axes);
            case IR.NTT.VectorizedBinary vectorizedBinary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, (Expr)arguments[2], vectorizedBinary.BinaryOp, vectorizedBinary.LhsVectorizedAxes, vectorizedBinary.LhsPadedNums, vectorizedBinary.RhsVectorizedAxes, vectorizedBinary.RhsPadedNums);
            case IR.NTT.VectorizedMatMul vectorizedMatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                var dinfo = vectorizedMatMul.GetDimInfo(dta.TensorType.Shape.Rank, dtb.TensorType.Shape.Rank);
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[dinfo.Lk] == dtb.AxisPolicies[dinfo.Rn] &&
                    dta.AxisPolicies[dinfo.Lm] == dtb.AxisPolicies[dinfo.Rk])
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB, vectorizedMatMul.FusedReduce);
                }

            case IR.Math.MatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[^2] == dtb.AxisPolicies[^2] &&
                    dta.AxisPolicies[^1] == dtb.AxisPolicies[^1])
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }

            case IR.CustomNTT.MatMul matmul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.CustomNTT.MatMul.Scale], matmul.LhsVectorizedAxes, matmul.RhsVectorizedAxes, matmul.TransposeA, matmul.TransposeB, false, matmul.CSourcePath, matmul.FuncName);
            case IR.NTT.PackedMatMul matmul:
                return TIR.F.NTT.PackedMatMul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.PackedMatMul.Scale], matmul.FusedReduce);
            case IR.NN.Conv2D conv:
                {
                    var input = call[IR.NN.Conv2D.Input];
                    var weights = call[IR.NN.Conv2D.Weights];
                    var bias = call[IR.NN.Conv2D.Bias];
                    var strides = ((RankedShape)call[IR.NN.Conv2D.Stride]).ToValueArray();
                    var padding = Tensor.From(((Paddings)call[IR.NN.Conv2D.Padding]).ToValueArray()).ToArray();
                    var dilation = ((RankedShape)call[IR.NN.Conv2D.Dilation]).ToValueArray();
                    var groups = ((Dimension)call[IR.NN.Conv2D.Groups]).FixedValue;
                    var fusedClamp = ((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = call.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    return TIR.F.NTT.Conv2D((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, strides, padding, dilation, groups, conv.PadMode, call.CheckedType is DistributedType dt_conv ? dt_conv : null!);
                }

            case IR.NTT.Im2col im2col:
                return TIR.F.NTT.Im2col((Expr)arguments[0], output, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.VectorizedAxes, im2col.PadedNums);
            case IR.NTT.VectorizedRoPE rope:
                return TIR.F.NTT.RoPE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Imaging.ResizeImage resize:
                if ((call[IR.Imaging.ResizeImage.Roi] is not None && ((RankedShape)call[IR.Imaging.ResizeImage.Roi].CheckedShape).Size != 0) || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Imaging.ResizeImage.NewSize]).ToValueArray().ToInts(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.NTT.ResizeImage resize:
                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, resize.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.ResizeImage.PadedNums]).Dimensions.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.Tensors.Slice slice:
                return TIR.F.NTT.Slice((Expr)arguments[0], (RankedShape)arguments[1], (RankedShape)arguments[2], output, ((RankedShape)call[IR.Tensors.Slice.Axes]).ToValueArray().ToInts(), ((RankedShape)call[IR.Tensors.Slice.Strides]).ToValueArray().ToInts());
            case IR.Tensors.Concat concat:
                return TIR.F.NTT.Concat(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, concat.Axis);
            case IR.Tensors.Transpose trans:
                return TIR.F.NTT.Transpose((Expr)arguments[0], output, ((RankedShape)call[IR.Tensors.Transpose.Perm]).ToValueArray().ToInts());
            case IR.NN.Swish swish:
                return TIR.F.NTT.Swish((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Swish.Beta]).Value.ToScalar<float>());
            case IR.Tensors.Gather gather:
                return TIR.F.NTT.Gather((Expr)arguments[0], (Expr)arguments[1], output, gather.Axis);
            case IR.NN.Pad pad:
                var paddings = (Paddings)call[IR.NN.Pad.Pads];
                var actualPadAxes = Enumerable.Range(0, paddings.Count).Where(i => !(paddings[i] is { IsFixed: true } pad && pad.Sum() == 0)).ToArray();
                return TIR.F.NTT.Pad((Expr)arguments[0], output, paddings, ((TensorConst)call[IR.NN.Pad.Value]).Value.ToArray<float>()[0], actualPadAxes);
            case IR.Math.Reduce reduce:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Math.Reduce.Axes]).ToValueArray().Select(x => Util.PositiveIndex(x, arguments[0].CheckedTensorType)).OrderBy(a => a).ToArray().ToInts(), ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToArray<bool>()[0], reduce.ReduceOp);
            case IR.Math.ReduceArg reduceArg:
                return TIR.F.NTT.ReduceArg((Expr)arguments[0], output, (int)((DimConst)call[IR.Math.ReduceArg.Axis]).FixedValue, ((TensorConst)call[IR.Math.ReduceArg.KeepDims]).Value.ToArray<bool>()[0], ((TensorConst)call[IR.Math.ReduceArg.SelectLastIndex]).Value.ToArray<bool>()[0], reduceArg.ReduceArgOp, reduceArg.DestType);
            case IR.Tensors.Cast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, Array.Empty<int>(), None.Default);
            case IR.NTT.VectorizedCast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, cast.VectorizeAxes, (Expr)arguments[1]);
            case IR.Tensors.Where where:
                return TIR.F.NTT.Where((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Expand expand:
                return TIR.F.NTT.Expand((Expr)arguments[0], output);
            case IR.NN.Erf erf:
                return TIR.F.NTT.Erf((Expr)arguments[0], output);
            case IR.NTT.VectorizedReduce pr:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, pr.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.VectorizedReduce.PadedNums]).Dimensions.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.NTT.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.NTT.GetItem((Expr)arguments[0], arguments[1], output);
            case IR.Tensors.Reshape:
                return GenerateReshape((Expr)arguments[0], ref output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.NTT.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Stack stack:
                return TIR.F.NTT.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
            case IR.Tensors.Unsqueeze:
                return GenerateReshape((Expr)arguments[0], ref output);
            case IR.NN.UpdatePagedAttentionKVCache upkv:
                output = (Expr)arguments[1];
                return TIR.F.NTT.UpdatePagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], upkv.CacheKind, upkv.LayerId, upkv.Layout);
            case IR.NN.GatherPagedAttentionKVCache gakv:
                return TIR.F.NTT.GatherPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], output);
            case IR.NN.CreatePagedAttentionKVCache ctkv:
                return TIR.F.NTT.CreatePagedAttentionKVCache(ctkv.Config, (Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], output);
            case IR.NN.IdentityPagedAttentionKVCache ctkv:
                output = (Expr)arguments[0];
                return TIR.F.NTT.IdentityPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8]);
            case IR.NN.PagedAttention pgat:
                return TIR.F.NTT.PagedAttention((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], pgat.LayerId, output, pgat.Layout, pgat.HiddenSize);
            case IR.Tensors.ConstantOfShape constantOfShape:
                return TIR.F.NTT.ConstantOfShape((Shape)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.Range range:
                return TIR.F.NTT.Range((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Buffers.Uninitialized uninitialized:
                return T.Nop();
            case IR.Shapes.AsTensor asTensor:
                output = call;
                return call;
            case IR.NN.GetPositionIds getPositionIds:
                return TIR.F.NTT.GetPositionIds((Expr)arguments[1], output, (DistributedType)call.CheckedType);
            case IR.NN.LayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.NTT.VectorizedLayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.VectorizedAxes, ((RankedShape)call[IR.NTT.VectorizedLayerNorm.PadedNums]).Dimensions.ToArray());
            case IR.NN.Softmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, (int)((DimConst)call[IR.NN.Softmax.Axis]).FixedValue, Array.Empty<int>());
            case IR.NTT.VectorizedSoftmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, softmax.Axis, softmax.VectorizedAxes);
            default:
                throw new NotSupportedException($"Not supported: {op}");
        }
    }

    private Expr GenerateReshape(Expr input, ref Expr output, bool sequeeze = false)
    {
        if (input is not TIR.Buffer inBuffer)
        {
            throw new NotSupportedException("Reshape only support buffer input");
        }

        var outBuffer = (TIR.Buffer)output;

        var bitcast = outBuffer.DistributedType is DistributedType ? false : true;
        if (!bitcast)
        {
            if (inBuffer.DistributedType!.AxisPolicies.Where(sbp => sbp is not SBPBroadCast).ToArray().SequenceEqual(outBuffer.DistributedType!.AxisPolicies.Where(sbp => sbp is not SBPBroadCast).ToArray()))
            {
                bitcast = true;
            }
        }

        // If the size is not same, we cannot bitcast.
        if ((inBuffer.MemSpan.Size == outBuffer.MemSpan.Size) && (bitcast || sequeeze))
        {
            output = inBuffer.With(name: outBuffer.Name, elemType: outBuffer.ElemType, dimensions: outBuffer.Dimensions.ToArray(), strides: outBuffer.Strides.ToArray(), distributedType: outBuffer.DistributedType);
            return T.Nop();
        }
        else
        {
            return TIR.F.NTT.Reshape(input, output);
        }
    }

    private Expr GenerateBitcast(Expr input, ref Expr output, DataType newType)
    {
        if (input is not TIR.Buffer inBuffer)
        {
            throw new NotSupportedException("Bitcast only support buffer input");
        }

        var srcSize = inBuffer.ElemType.SizeInBytes;
        var destSize = newType.SizeInBytes;
        var newDimensions = inBuffer.Dimensions.ToArray();
        var newStrides = inBuffer.Strides.ToArray();

        if (srcSize != destSize)
        {
            if (newDimensions.Rank == 0)
            {
                newDimensions = [srcSize / destSize];
                newStrides = [1];
            }
            else
            {
                newDimensions[^1] = newDimensions[^1] * srcSize / destSize;
                if (newStrides.Length > 1)
                {
                    newStrides[^1] = 1;
                    for (var i = 0; i < newStrides.Length - 1; i++)
                    {
                        newStrides[i] = newStrides[i] * srcSize / destSize;
                    }
                }
            }
        }

        var distributedType = inBuffer.DistributedType is DistributedType dt
            ? dt with { TensorType = new TensorType(newType, newDimensions) }
            : null;
        output = inBuffer.With(name: ((TIR.Buffer)output).Name, elemType: newType, dimensions: newDimensions, strides: newStrides, distributedType: distributedType);
        return T.Nop();
    }

    private Expr GenerateUnary(UnaryOp unaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var input = (Expr)arguments[IR.Math.Unary.Input.Index];
        return TIR.F.NTT.Unary(unaryOp, input, output);
    }

    private Expr GenerateClamp(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var min = ((TensorConst)call[IR.Math.Clamp.Min]).Value.ToScalar<float>();
        var max = ((TensorConst)call[IR.Math.Clamp.Max]).Value.ToScalar<float>();
        return TIR.F.NTT.Clamp((Expr)arguments[0], output, min, max);
    }

    private Expr GenerateBoxing(Call call, IR.Distributed.Boxing boxing, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        switch (call[IR.Distributed.Boxing.Input].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                return TIR.F.NTT.TensorLoad(output, (Expr)arguments[0], distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType distTensorType, TensorType):
                return TIR.F.NTT.TensorStore((Expr)arguments[0], output, distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType inType, DistributedType outType):
                return GenerateReshard((Expr)arguments[0], ref output, inType, outType);
            default:
                throw new NotSupportedException();
        }
    }

    private Expr GenerateReshard(Expr input, ref Expr output, DistributedType inType, DistributedType outType)
    {
        if (input is TIR.Buffer inBuffer)
        {
            if (TryGenerateGatherThreadsReshard(inBuffer, ref output, inType, outType, out var newCall))
            {
                return newCall;
            }
            else if (TryGenerateSplitThreadsReshard(inBuffer, ref output, inType, outType, out newCall))
            {
                return newCall;
            }
        }

        return TIR.F.NTT.GatherReduceScatter(input, output, inType, outType);
    }

    private bool TryGenerateGatherThreadsReshard(TIR.Buffer inBuffer, ref Expr output, DistributedType inType, DistributedType outType, [MaybeNullWhen(false)] out Expr newCall)
    {
        var threadAxis = inType.Placement.Rank - 1;
        PhysicalBuffer? oldPhysicalBuffer = null;

        // S -> B
        var reducedInPolices = inType.AxisPolicies.Select(sbp => sbp is SBPSplit split && split.Axes.Contains(threadAxis) ? (split.Axes.Count == 1 ? (SBP)SBP.B : SBP.S(split.Axes.Except([threadAxis]).ToArray())) : sbp);
        if (reducedInPolices.ToArray().SequenceEqual(outType.AxisPolicies.ToArray()))
        {
            oldPhysicalBuffer = inBuffer.MemSpan.Buffer;
        }

        var oldOutputBuffer = (TIR.Buffer)output;
        if (oldPhysicalBuffer is not null)
        {
            var threads = inType.Placement.Hierarchy[threadAxis];
            var newPhysicalBuffer = oldPhysicalBuffer.With(
                size: oldOutputBuffer.MemSpan.Size,
                location: MemoryLocation.BlockLocalData);

            // 1. Replace all uses of old buffer to new buffer.
            var userBuffers = (from memSpan in oldPhysicalBuffer.Users.OfType<TIR.MemSpan>()
                               from userBuffer in memSpan.Users.OfType<TIR.Buffer>()
                               select userBuffer).ToArray();

            foreach (var userBuffer in userBuffers)
            {
                var userType = userBuffer.DistributedType!;
                var threadAxisDim = userType.AxisPolicies.ToArray().IndexOf(x => x is SBPSplit split && split.Axes.Contains(threadAxis));
                if (threadAxisDim >= 0)
                {
                    var newStrides = userBuffer.Strides.ToArray();
                    for (int i = 0; i < userType.AxisPolicies.Count; i++)
                    {
                        if (i < threadAxisDim)
                        {
                            newStrides[i] *= threads;
                        }

                        if (i == threadAxisDim)
                        {
                            var newStart = userBuffer.MemSpan.Start;
                            if (newStart != Dimension.Zero)
                            {
                                // We don't support sliced buffer.
                                newCall = null;
                                return false;
                            }

                            var dividedType = DistributedUtility.GetDividedTensorType(userType);
                            newStart = dividedType.Shape[i] * newStrides[i] * userBuffer.CheckedDataType.SizeInBytes * IR.F.Distributed.ThreadId();
                            var newBuffer = userBuffer.With(
                                memSpan: userBuffer.MemSpan.With(
                                    buffer: newPhysicalBuffer,
                                    start: newStart),
                                strides: newStrides);
                            ReplaceUtility.ReplaceAllUsesWith(userBuffer, newBuffer);
                            break;
                        }
                    }
                }
            }

            // 2. Create new output buffer.
            output = oldOutputBuffer.With(
                memSpan: oldOutputBuffer.MemSpan.With(buffer: newPhysicalBuffer));
            newCall = TIR.F.NTT.SynchronizeThreads();
            return true;
        }

        newCall = null;
        return false;
    }

    private bool TryGenerateSplitThreadsReshard(TIR.Buffer inBuffer, ref Expr output, DistributedType inType, DistributedType outType, [MaybeNullWhen(false)] out Expr newCall)
    {
        var threadAxis = inType.Placement.Rank - 1;
        PhysicalBuffer? oldPhysicalBuffer = null;
        var reducedOutPolices = outType.AxisPolicies.Select(sbp => sbp is SBPSplit split && split.Axes.Contains(threadAxis) ? (split.Axes.Count == 1 ? (SBP)SBP.B : SBP.S(split.Axes.Except([threadAxis]).ToArray())) : sbp);
        int splitAxis = -1;

        // B -> S
        if (reducedOutPolices.ToArray().SequenceEqual(inType.AxisPolicies.ToArray()))
        {
            oldPhysicalBuffer = inBuffer.MemSpan.Buffer;
            splitAxis = outType.AxisPolicies.ToArray().IndexOf(x => x is SBPSplit split && split.Axes.Contains(threadAxis));
        }

        var oldOutputBuffer = (TIR.Buffer)output;
        if (oldPhysicalBuffer is not null)
        {
            var threads = inType.Placement.Hierarchy[threadAxis];
            var newPhysicalBuffer = oldPhysicalBuffer.With(
                location: MemoryLocation.BlockLocalData);

            // 1. Replace all uses of old mem span to new mem span.
            var userMemSpans = oldPhysicalBuffer.Users.OfType<TIR.MemSpan>().ToArray();

            foreach (var userMemSpan in userMemSpans)
            {
                var newMemSpan = userMemSpan.With(buffer: newPhysicalBuffer);
                ReplaceUtility.ReplaceAllUsesWith(userMemSpan, newMemSpan);
            }

            // 2. Create new output buffer.
            var newStart = oldOutputBuffer.MemSpan.Start;
            if (newStart != Dimension.Zero)
            {
                // We don't support sliced buffer.
                newCall = null;
                return false;
            }

            var dividedType = DistributedUtility.GetDividedTensorType(outType);
            newStart = dividedType.Shape[splitAxis] * oldOutputBuffer.Strides[splitAxis] * oldOutputBuffer.CheckedDataType.SizeInBytes * IR.F.Distributed.ThreadId();

            output = oldOutputBuffer.With(
                memSpan: oldOutputBuffer.MemSpan.With(buffer: newPhysicalBuffer, start: newStart), strides: inBuffer.Strides.ToArray());
            newCall = TIR.F.NTT.SynchronizeThreads();
            return true;
        }

        newCall = null;
        return false;
    }
}
