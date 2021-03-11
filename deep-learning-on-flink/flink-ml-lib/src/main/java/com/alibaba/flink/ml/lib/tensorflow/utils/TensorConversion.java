/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.alibaba.flink.ml.lib.tensorflow.utils;

import com.google.common.base.Preconditions;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.StdArrays;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;

import java.io.Serializable;


// TODO: support more kinds of tensors

/**
 * a util function: transformation between tensorflow tensor and java object.
 */
public class TensorConversion {

	private TensorConversion() {
	}

	/**
	 * convert java objects to tensorflow tensor.
	 * @param object given java objects.
	 * @param tensorInfo target tensor type information.
	 * @return result tensorflow tensor.
	 */
	public static Tensor<?> toTensor(Object object, TensorInfo tensorInfo, int rank) {

		switch (tensorInfo.getDtype()) {
			case DT_INT32: {
				switch (rank){
					case 1:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[]) object));
					}
					case 2:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[][]) object));
					}
					case 3:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[][][]) object));
					}
					case 4:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[][][][]) object));
					}
					case 5:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[][][][][]) object));
					}
					case 6:{
						return TInt32.tensorOf(StdArrays.ndCopyOf((int[][][][][][]) object));
					}
					default:
						throw new UnsupportedOperationException(
								"dim count can't supported: " + rank);
				}
			}
			case DT_INT64: {
				switch (rank){
					case 1:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[])object));
					}
					case 2:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[][])object));
					}
					case 3:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[][][])object));
					}
					case 4:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[][][][])object));
					}
					case 5:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[][][][][])object));
					}
					case 6:{
						return TInt64.tensorOf(StdArrays.ndCopyOf((long[][][][][][])object));
					}
					default:
						throw new UnsupportedOperationException("dim count can't supported: " + rank);
				}
			}
			case DT_FLOAT: {
				switch (rank){
					case 1:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[])object));
					}
					case 2:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[][])object));
					}
					case 3:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[][][])object));
					}
					case 4:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[][][][])object));
					}
					case 5:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[][][][][])object));
					}
					case 6:{
						return TFloat32.tensorOf(StdArrays.ndCopyOf((float[][][][][][])object));
					}
					default:
						throw new UnsupportedOperationException("dim count can't supported: " + rank);
				}
			}
			case DT_DOUBLE: {
				switch (rank){
					case 1:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[])object));
					}
					case 2:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[][])object));
					}
					case 3:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[][][])object));
					}
					case 4:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[][][][])object));
					}
					case 5:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[][][][][])object));
					}
					case 6:{
						return TFloat64.tensorOf(StdArrays.ndCopyOf((double[][][][][][])object));
					}
					default:
						throw new UnsupportedOperationException("dim count can't supported: " + rank);
				}
			}
			case DT_STRING: {
				switch (rank){
					case 1:{
						String[] stringArray = (String[])object;
						byte[][] temp = new byte[stringArray.length][];
						for(int i = 0; i < stringArray.length; i++){
							temp[i] = stringArray[i].getBytes();
						}
						return TUint8.tensorOf(StdArrays.ndCopyOf(temp));
					}
					case 2:{
						String[][] stringArray = (String[][])object;
						int len1 = stringArray.length;
						int len2 = stringArray[0].length;
						byte[][][] temp = new byte[len1][len2][];
						for(int i = 0; i < len1; i++){
							for(int j = 0; j < len2; j++) {
								temp[i][j] = stringArray[i][j].getBytes();
							}
						}
						return TUint8.tensorOf(StdArrays.ndCopyOf(temp));
					}
					case 3:{
						String[][][] stringArray = (String[][][])object;
						int len1 = stringArray.length;
						int len2 = stringArray[0].length;
						int len3 = stringArray[0][0].length;
						byte[][][][] temp = new byte[len1][len2][len3][];
						for(int i = 0; i < len1; i++){
							for(int j = 0; j < len2; j++) {
								for(int k = 0; k < len3; k++) {
									temp[i][j][k] = stringArray[i][j][k].getBytes();
								}
							}
						}
						return TUint8.tensorOf(StdArrays.ndCopyOf(temp));
					}
					case 4:{
						String[][][][] stringArray = (String[][][][])object;
						int len1 = stringArray.length;
						int len2 = stringArray[0].length;
						int len3 = stringArray[0][0].length;
						int len4 = stringArray[0][0][0].length;
						byte[][][][][] temp = new byte[len1][len2][len3][len4][];
						for(int i = 0; i < len1; i++){
							for(int j = 0; j < len2; j++) {
								for(int k = 0; k < len3; k++) {
									for(int m = 0; m < len4; m++) {
										temp[i][j][k][m] = stringArray[i][j][k][m].getBytes();
									}
								}
							}
						}
						return TUint8.tensorOf(StdArrays.ndCopyOf(temp));
					}
					case 5:{
						String[][][][][] stringArray = (String[][][][][])object;
						int len1 = stringArray.length;
						int len2 = stringArray[0].length;
						int len3 = stringArray[0][0].length;
						int len4 = stringArray[0][0][0].length;
						int len5 = stringArray[0][0][0][0].length;

						byte[][][][][][] temp = new byte[len1][len2][len3][len4][len5][];
						for(int i = 0; i < len1; i++){
							for(int j = 0; j < len2; j++) {
								for(int k = 0; k < len3; k++) {
									for(int m = 0; m < len4; m++) {
										for(int n = 0; n < len5; n++) {
											temp[i][j][k][m][n] = stringArray[i][j][k][m][n].getBytes();
										}
									}
								}
							}
						}
						return TUint8.tensorOf(StdArrays.ndCopyOf(temp));
					}
					default:
						throw new UnsupportedOperationException("dim count can't supported: " + rank);
				}
			}
			default:
				throw new UnsupportedOperationException(
						"Type can't be converted to tensor: " + tensorInfo.getDtype().name());
		}
	}


	/**
	 * convert tensorflow tensor to java objects.
	 * @param tensor given tensorflow tensor.
	 * @return java objects corresponded to given tensor.
	 */
	public static Object fromTensor(Tensor<?> tensor) {
		final double numDimensions = tensor.shape().numDimensions();
		Preconditions.checkArgument(numDimensions <= 5,
				"Can only convert tensors with shape less than 2 current " + numDimensions);
		final int rank = (int) numDimensions;
		if (TInt32.DTYPE.equals(tensor.dataType())) {
			return fromIntTensor(tensor, rank);
		} else if (TFloat32.DTYPE.equals(tensor.dataType())) {
			return fromFloatTensor(tensor, rank);
		} else if (TInt64.DTYPE.equals(tensor.dataType())) {
			return fromLongTensor(tensor, rank);
		} else if (TFloat64.DTYPE.equals(tensor.dataType())) {
			return fromDoubleTensor(tensor, rank);
		} else if (TUint8.DTYPE.equals(tensor.dataType())) {
			return fromStringTensor(tensor, rank);
		}

		throw new UnsupportedOperationException(
				"Type can't be converted from tensor : " + tensor.dataType().name());
	}

	private static Serializable[] fromStringTensor(Tensor<?> tensor, int rank) {
		switch (rank) {
			case 1: {
				final int len1 = (int) tensor.shape().size(0);
				final byte[][] resultBytes = new byte[len1][];
				StdArrays.copyFrom((TUint8) tensor.data(), resultBytes);
				String[] results = new String[len1];
				for (int i = 0; i < len1; i++){
					results[i] = new String(resultBytes[i]);
				}
				return results;
			}
			case 2: {
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final byte[][][] resultBytes = new byte[len1][len2][];
				StdArrays.copyFrom((TUint8) tensor.data(), resultBytes);
				String[][] results = new String[len1][len2];
				for (int i = 0; i < len1; i++) {
					for (int j = 0; j < len2; j++) {
						results[i][j] = new String(resultBytes[i][j]);
					}
				}
				return results;
			}
			case 3: {
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final byte[][][][] resultBytes = new byte[len1][len2][len3][];
				StdArrays.copyFrom((TUint8) tensor.data(), resultBytes);
				String[][][] results = new String[len1][len2][len3];
				for (int i = 0; i < len1; i++) {
					for (int j = 0; j < len2; j++) {
						for (int k = 0; k < len3; k++) {
							results[i][j][k] = new String(resultBytes[i][j][k]);
						}
					}
				}
				return results;
			}
			case 4: {
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final byte[][][][][] resultBytes = new byte[len1][len2][len3][len4][];
				StdArrays.copyFrom((TUint8) tensor.data(), resultBytes);
				String[][][][] results = new String[len1][len2][len3][len4];
				for (int i = 0; i < len1; i++) {
					for (int j = 0; j < len2; j++) {
						for (int k = 0; k < len3; k++) {
							for (int m = 0; m < len4; m++) {
								results[i][j][k][m] = new String(resultBytes[i][j][k][m]);
							}
						}
					}
				}
				return results;
			}
			case 5: {
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int len5 = (int) tensor.shape().size(4);
				final byte[][][][][][] resultBytes = new byte[len1][len2][len3][len4][len5][];
				StdArrays.copyFrom((TUint8) tensor.data(), resultBytes);
				String[][][][][] results = new String[len1][len2][len3][len4][len5];
				for (int i = 0; i < len1; i++) {
					for (int j = 0; j < len2; j++) {
						for (int k = 0; k < len3; k++) {
							for (int m = 0; m < len4; m++) {
								for (int n = 0; n < len5; n++) {
									results[i][j][k][m][n] = new String(resultBytes[i][j][k][m][n]);
								}
							}
						}
					}
				}
				return results;
			}
			default:
				throw new UnsupportedOperationException(
						"dim count can't supported: " + rank);
		}
	}

	private static Cloneable fromDoubleTensor(Tensor<?> tensor, int rank) {
		switch (rank){
			case 1:{
				final int len1 = (int) tensor.shape().size(0);
				final double[] res = new double[len1];
				StdArrays.copyFrom((TFloat64) tensor.data(), res);
				return res;
			}
			case 2:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final double[][] res = new double[len1][len2];
				StdArrays.copyFrom((TFloat64) tensor.data(), res);
				return res;
			}
			case 3:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final double[][][] res = new double[len1][len2][len3];
				StdArrays.copyFrom((TFloat64) tensor.data(), res);
				return res;
			}
			case 4:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final double[][][][] res = new double[len1][len2][len3][len4];
				StdArrays.copyFrom((TFloat64) tensor.data(), res);
				return res;
			}
			case 5:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int len5 = (int) tensor.shape().size(4);
				final double[][][][][] res = new double[len1][len2][len3][len4][len5];
				StdArrays.copyFrom((TFloat64) tensor.data(), res);
				return res;
			}
			default:
				throw new UnsupportedOperationException(
						"dim count can't supported: " + rank);
		}
	}

	private static Cloneable fromLongTensor(Tensor<?> tensor, int rank) {
		switch (rank){
			case 1:{
				final int len1 = (int) tensor.shape().size(0);
				final long[] res = new long[len1];
				StdArrays.copyFrom((TInt64) tensor.data(), res);
				return res;
			}
			case 2:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final long[][] res = new long[len1][len2];
				StdArrays.copyFrom((TInt64) tensor.data(), res);
				return res;
			}
			case 3:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final long[][][] res = new long[len1][len2][len3];
				StdArrays.copyFrom((TInt64) tensor.data(), res);
				return res;
			}
			case 4:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final long[][][][] res = new long[len1][len2][len3][len4];
				StdArrays.copyFrom((TInt64) tensor.data(), res);
				return res;
			}
			case 5:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int len5 = (int) tensor.shape().size(4);
				final long[][][][][] res = new long[len1][len2][len3][len4][len5];
				StdArrays.copyFrom((TInt64) tensor.data(), res);
				return res;
			}
			default:
				throw new UnsupportedOperationException(
						"dim count can't supported: " + rank);
		}
	}

	private static Cloneable fromFloatTensor(Tensor<?> tensor, int rank) {
		switch (rank){
			case 1:{
				final int len1 = (int) tensor.shape().size(0);
				final float[] res = new float[len1];
				StdArrays.copyFrom((TFloat32) tensor.data(), res);
				return res;
			}
			case 2:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final float[][] res = new float[len1][len2];
				StdArrays.copyFrom((TFloat32) tensor.data(), res);
				return res;
			}
			case 3:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final float[][][] res = new float[len1][len2][len3];
				StdArrays.copyFrom((TFloat32) tensor.data(), res);
				return res;
			}
			case 4:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final float[][][][] res = new float[len1][len2][len3][len4];
				StdArrays.copyFrom((TFloat32) tensor.data(), res);
				return res;
			}
			case 5:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int len5 = (int) tensor.shape().size(4);
				final float[][][][][] res = new float[len1][len2][len3][len4][len5];
				StdArrays.copyFrom((TFloat32) tensor.data(), res);
				return res;
			}
			default:
				throw new UnsupportedOperationException(
						"dim count can't supported: " + rank);
		}
	}

	private static Object fromIntTensor(Tensor<?> tensor, int rank) {
		switch (rank){
			case 1:{
				final int len1 = (int) tensor.shape().size(0);
				final int[] res = new int[len1];
				StdArrays.copyFrom((TInt32) tensor.data(), res);
				return res;
			}
			case 2:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int[][] res = new int[len1][len2];
				StdArrays.copyFrom((TInt32) tensor.data(), res);
				return res;
			}
			case 3:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int[][][] res = new int[len1][len2][len3];
				StdArrays.copyFrom((TInt32) tensor.data(), res);
				return res;
			}
			case 4:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int[][][][] res = new int[len1][len2][len3][len4];
				StdArrays.copyFrom((TInt32) tensor.data(), res);
				return res;
			}
			case 5:{
				final int len1 = (int) tensor.shape().size(0);
				final int len2 = (int) tensor.shape().size(1);
				final int len3 = (int) tensor.shape().size(2);
				final int len4 = (int) tensor.shape().size(3);
				final int len5 = (int) tensor.shape().size(4);
				final int[][][][][] res = new int[len1][len2][len3][len4][len5];
				StdArrays.copyFrom((TInt32) tensor.data(), res);
				return res;
			}
			default:
				throw new UnsupportedOperationException(
						"dim count can't supported: " + rank);
		}
	}

}
