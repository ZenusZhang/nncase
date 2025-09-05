﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;
using Nncase.TIR.NTT;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectRoPE(IR.NN.RoPE rope, Call call, Expr output)
    {
        var input = (Expr)call[IR.NN.RoPE.Input];
        var cos = (Expr)call[IR.NN.RoPE.Cos];
        var sin = (Expr)call[IR.NN.RoPE.Sin];
        var lastDim = input.CheckedShape[^1].FixedValue;

        var rank = input.CheckedShape.Rank;
        var domains = IR.F.Affine.Domains(rank);
        var inOutResults = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        var inOutMap = new AffineMap(domains, default, inOutResults);
        var sinCosResults = cos.CheckedShape.Rank == 2 ? inOutResults[1..] : [inOutResults[0], new AffineRange(0, 1), inOutResults[2]];
        var sinCosMap = new AffineMap(domains, default, sinCosResults);

        return IR.F.Affine.Grid()
            .Domain(rank, out var _)
            .Read(input, inOutMap, out var inTile)
            .Read(sin, sinCosMap, out var sinTile)
            .Read(cos, sinCosMap, out var cosTile)
            .Write(output, inOutMap, out var outTile)
            .Body(TIR.F.NTT.RoPE(inTile, cosTile, sinTile, outTile))
            .Build();
    }
}
