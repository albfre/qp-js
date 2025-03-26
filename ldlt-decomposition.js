function ldltDecomposition(A) {
  const n = A.length;
  let L = Array.from({ length: n }, () => Array(n).fill(0));
  let D = Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    let sumD = A[i][i];
    for (let j = 0; j < i; j++) {
      sumD -= L[i][j] * L[i][j] * D[j];
    }
    // Modify factorization to avoid zero diagonal (Vanderbei, Symmetric quasi-definite matrices, 1995)
    D[i] = sumD == 0.0 ? 1e-8 : sumD;

    for (let j = i + 1; j < n; j++) {
      let sum = A[j][i];
      for (let k = 0; k < i; k++) {
        sum -= L[j][k] * L[i][k] * D[k];
      }
      L[j][i] = sum / D[i];
    }

    L[i][i] = 1;
  }

  return [ L, D ];
}

function solveLDLT(L, D, b) {
  const n = L.length;
  let y = new Array(n).fill(0);

  // Forward substitution (L * y = b)
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = b[i] - sum;
  }

  // Solve for z (D * z = y)
  for (let i = 0; i < n; i++) {
    y[i] /= D[i];
  }

  // Backward substitution (L^T * y = z)
  for (let i = n - 1; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += L[j][i] * y[j];
    }
    y[i] -= sum;
  }

  return y;
}
