﻿//Extension of the OLS regression using dense matrices that performs a
//Cholesky decomposition
IMPORT ML;
IMPORT ML.Types AS Types;
IMPORT PBblas_v0 as PBblas_v0;
IMPORT ML.DMat as DMat;
IMPORT ML.Regression.Dense;
NotCompat := PBblas_v0.Constants.Dimension_Incompat;
Matrix_Map:= PBblas_v0.Matrix_Map;
LowerTri  := PBblas_v0.Types.Triangle.Lower;
UpperTri  := PBblas_v0.Types.Triangle.Upper;
NotUnit   := PBblas_v0.Types.Diagonal.NotUnitTri;
Side      := PBblas_v0.Types.Side;
Part      := PBblas_v0.Types.Layout_Part;
NumericField := Types.NumericField;

EXPORT OLS_Cholesky(DATASET(NumericField) X,DATASET(NumericField) Y)
:= MODULE(ML.Regression.Dense.OLS(X,Y))
  x2_map := Matrix_Map(x_rows, x_cols, block_rows, x_cols);
  y2_map := Matrix_Map(y_rows, y_cols, block_rows, y_cols);
  b2_map := Matrix_Map(x_cols, y_cols, x_cols, y_cols);
  z2_map := Matrix_Map(x_cols, x_cols, x_cols, x_cols);
  // Calculate the model beta matrix
  XtX_p := PBblas_v0.PB_dbvrk(TRUE, 1.0, x2_map, x_part, z2_map);
  XtY_p := PBblas_v0.PB_dbvmm(TRUE, FALSE, 1.0, x2_map, x_part, y2_map, y_part,
                          b2_map);
  L_p   := PBblas_v0.PB_dpotrf(LowerTri, z2_map, XtX_p);
  s1_p  := PBblas_v0.PB_dtrsm(Side.Ax, LowerTri, FALSE, NotUnit, 1.0,
                          z2_map, L_p, b2_map, XtY_p);
  b_part:= PBblas_v0.PB_dtrsm(Side.Ax, UpperTri, TRUE, NotUnit, 1.0,
                          z2_map, L_p, b2_map, s1_p);
  EXPORT DATASET(Part) BetasAsPartition := b_part;
END;
