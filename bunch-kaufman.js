function symmetricIndefiniteFactorization(Ain) {
  // Bunch-Kaufman factorization
  const A = Ain.map(row => [...row]);
  const n = A.length;
  const alpha = (1.0 + Math.sqrt(17)) / 8;
  const ipiv = zeroVector(n);

  let info = 0;

  let k = 0; // k is the main loop index, increasing from 1 to n in steps of 1 or 2
  while (k < n) {
    let kstep = 1
    let kp = 0;
    const absakk = Math.abs(A[k][k]);
    // imax is the row-index of the largest off-diagonal element in column k, and colmax is its absolute value
    let imax = 0;
    let colmax = 0.0;
    for (let i = k + 1; i < n; i++) {
      const v = Math.abs(A[i][k]);
      if (v > colmax) {
        colmax = v;
        imax = i;
      }
    }
    if (absakk === 0.0 && colmax === 0.0) {
      // Column k is zero: set info and continue
      if (info === 0) {
        info = k;
        kp = k;
      }
    }
    else {
      if (absakk >= alpha * colmax) {
        // no interchange, use 1-by-1 pivot block
        kp = k;
      }
      else {
        // jmax is the column-index of the largest off-diagonal element in row imax, and rowmax is its absolute value
        let rowmax = 0.0;
        let jmax = 0;
        for (let j = k; j < imax; j++) {
          const v = Math.abs(A[imax][j]);
          if (v > rowmax) {
            rowmax = v;
            jmax = j;
          }
        }
        for (let j = imax + 1; j < n; j++) {
          const v = Math.abs(A[j][imax]);
          if (v > rowmax) {
            rowmax = v;
            jmax = j;
          }
        }
        if (absakk * rowmax >= alpha * colmax * colmax) {
          // no interchange, use 1-by-1 pivot block
          kp = k
        }
        else if (Math.abs(A[imax][imax]) >= alpha * rowmax) {
          // interchange rows and columns k and imax, use 1-by-1 pivot block
          kp = imax;
        }
        else {
          // interchange rows and columns k+1 and imax, use 2-by-2 pivot block
          kp = imax;
          kstep = 2;
        }
      }
      const kk = k + kstep - 1;
      if (kp !== kk) {
        // Interchange rows and columns kk and kp in the trailing submatrix A(k:n,k:n)
        for (let i = k; i < n; i++) {
          [A[i][kp], A[i][kk]] = [A[i][kk], A[i][kp]];
        }
        for (let j = k; j < n; j++) {
          [A[kp][j], A[kk][j]] = [A[kk][j], A[kp][j]];
        }
      }
      // Update the trailing submatrix
      if (kstep === 1) {
        // 1-by-1 pivot block D(k): column k now holds W(k) = L(k)*D(k) where L(k) is the k-th column of L
        // Perform a rank-1 update of A(k+1:n,k+1:n) as A := A - L(k)*D(k)*L(k)**T = A - W(k)*(1/D(k))*W(k)**T
        const r1 = 1.0 / A[k][k];
        const row = A[k];
        for (let j = k + 1; j < n; j++) {
          for (let i = j; i < n; i++) {
            A[i][j] -= r1 * row[i] * row[j]
            A[j][i] = A[i][j];
          }
        }

        for (let i = k + 1; i < n; i++) {
          A[i][k] *= r1;
          A[k][i] = A[i][k];
        }
      }
      else {
        // 2-by-2 pivot block D(k): columns k and k+1 now hold ( W(k) W(k+1) ) = ( L(k) L(k+1) )*D(k)
        // where L(k) and L(k+1) are the k-th and (k+1)-th columns of L
        if (k < n - 1) {
          // Perform a rank-2 update of A(k+2:n,k+2:n) as
          // A := A - ( L(k) L(k+1) )*D(k)*( L(k) L(k+1) )**T = A - ( W(k) W(k+1) )*inv(D(k))*( W(k) W(k+1) )**T
          // where L(k) and L(k+1) are the k-th and (k+1)-th columns of L

          let d21 = A[k + 1][k];
          const d11 = A[k + 1][k + 1] / d21;
          const d22 = A[k][k] / d21;
          const t = 1.0 / (d11 * d22 - 1.0);
          d21 = t / d21;

          for (let j = k + 2; j < n; j++) {
            const wk = d21 * (d11 * A[j][k] - A[j][k + 1]);
            const wkp1 = d21 * (d22 * A[j][k + 1] - A[j][k]);
            for (let i = j; i < n; i++) {
              A[i][j] -= (A[i][k] * wk + A[i][k + 1] * wkp1);
              A[j][i] = A[i][j];
            }
            A[j][k] = wk;
            A[j][k + 1] = wkp1;
            A[k][j] = A[j][k];
            A[k + 1][j] = A[j][k + 1];
          }
        }
      }
    }
    // Store details of the interchanges in ipiv
    if (kstep === 1) {
      ipiv[k] = kp;
    }
    else {
      ipiv[k] = -kp;
      ipiv[k + 1] = -kp;
    }
    k += kstep;
  }

  return [A, ipiv];
}

function solveUsingFactorization(L, ipiv, bin) {
  // Solve A*X = B, where A = L*D*L**T.
  const b = [...bin];
  assertIsMatrix(L);
  L.every(row => assertAreEqualLengthVectors(row, b));
  assertAreEqualLengthVectors(ipiv, b);
  const n = b.length;

  // First solve L*D*X = B, overwriting B with X.
  // k is the main loop index, increasing from 1 to n in steps of 1 or 2, depending on the size of the diagonal blocks.
  function dger(i_start, j_index) {
    const temp = -b[j_index];
    for (let i = i_start; i < n; i++) {
      b[i] += L[i][j_index] * temp;
    }
  }
  let k = 0;
  while (k < n) {
    if (ipiv[k] >= 0) {
      // 1 x 1 diagonal block, interchange rows k and ipiv(k).
      const kp = ipiv[k];
      if (kp != k) {
        [b[k], b[kp]] = [b[kp], b[k]];
      }
      // Multiply by inv(L(k)), where L(k) is the transformation stored in column k of L.
      dger(k + 1, k);
      b[k] /= L[k][k];
      k += 1;
    }
    else {
      // 2 x 2 diagonal block, interchange rows k+1 and -ipiv(k).

      const kp = -ipiv[k];
      if (kp !== k + 1) {
        [b[k + 1], b[kp]] = [b[kp], b[k + 1]];
      }
      // Multiply by inv(L(k)), where L(k) is the transformation stored in columns k and k+1 of L.
      if (k < n - 1) {
        dger(k + 2, k);
        dger(k + 2, k + 1);
      }
      // Multiply by the inverse of the diagonal block.
      const akm1k = L[k + 1][k];
      const akm1 = L[k][k] / akm1k;
      const ak = L[k + 1][k + 1] / akm1k;
      const denom = akm1 * ak - 1.0;
      const bkm1 = b[k] / akm1k;
      const bk = b[k + 1] / akm1k;
      b[k] = (ak * bkm1 - bk) / denom;
      b[k + 1] = (akm1 * bk - bkm1) / denom;
      k = k + 2
    }
  }

  // Next solve L**T *X = B, overwriting B with X.
  // k is the main loop index, decreasing from n - 1 to 0 in steps of 1 or 2, depending on the size of the diagonal blocks.
  function dgemv(i_start, j_index) {
    let temp = 0.0;
    for (let i = i_start; i < n; ++i) {
      temp += L[i][j_index] * b[i];
    }
    b[j_index] -= temp;
  }
  k = n - 1;
  while (k >= 0) {
    if (ipiv[k] >= 0) {
      // 1 x 1 diagonal block, multiply by inv(L**T(k)), where L(k) is the transformation stored in column k of L.
      if (k < n - 1) {
        dgemv(k + 1, k); // Subroutine dgemv 'Transpose' with alpha = -1 and beta = 1
      }

      // Interchange rows K and IPIV(K).
      const kp = ipiv[k];
      if (kp !== k) {
        [b[k], b[kp]] = [b[kp], b[k]];
      }

      k -= 1;
    }
    else {
      // 2 x 2 diagonal block, multiply by inv(L**T(k-1)), where L(k-1) is the transformation stored in columns k-1 and k of L.
      if (k < n - 1) {
        dgemv(k + 1, k);
        dgemv(k + 1, k - 1);
      }

      // Interchange rows k and -ipiv(k).
      const kp = -ipiv[k];
      if (kp !== k) {
        [b[k], b[kp]] = [b[kp], b[k]];
      }

      k -= 2;
    }
  }
  return b;
}

function symmetricIndefiniteFactorization_unstable(A) {
  const n = A.length;
  const L = zeroMatrix(n, n);
  const D = zeroVector(n);
  const P = zeroVector(n).map((_, i) => i);

  for (let i = 0; i < n; i++) {
    // Compute the (i,i) entry of D
    let d_ii = A[i][i];
    for (let k = 0; k < i; k++) {
      d_ii -= L[i][k] ** 2 * D[k];
    }

    // Check for singularity
    if (d_ii === 0) {
      throw new Error('Matrix is singular');
    }

    D[i] = d_ii;

    // Compute the entries of L
    for (let j = i + 1; j < n; j++) {
      let l_ij = A[i][j];
      for (let k = 0; k < i; k++) {
        l_ij -= L[i][k] * D[k] * L[j][k];
      }
      L[j][i] = l_ij / d_ii;
    }
  }

  return [L, D, P ];
}

function solveUsingFactorization_unstable(L, D, b) {
  const n = L.length;
  const x = zeroVector(n);
  const y = zeroVector(n);

  // Forward substitution: solve Ly = b
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }

  // Backward substitution: solve L^Tx = y
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += L[j][i] * x[j];
    }
    x[i] = (y[i] - sum) / D[i];
  }

  return x;
}
