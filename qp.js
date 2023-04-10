function interiorPointQP(H, c, Aeq, beq, Aineq, bineq, tol=1e-8, maxIter=100) {
  /* minimize 0.5 x' H x + c' x
   *   st    Aeq x = beq
   *         Aineq x >= bineq
   */

  // Matrix sizes
  const n = H.length;
  const mIneq = Aineq.length;
  const mEq = Aeq.length;

  // Preconditions
  if (H.some(row => row.length != n)) {
    throw new Error('H is not a square matrix');
  }
  if (Aineq.some(row => row.length != n)) {
    throw new Error('All rows of Aineq must have the same length as H');
  }
  if (Aeq.some(row => row.length != n)) {
    throw new Error('All rows of Aeq must have the same length as H');
  }
  if (bineq.length !== mIneq) {
    throw new Error('Aineq and bineq must have the same length. Aineq.length = ' + mIneq + ', bineq.length = ' + bineq.length);
  }
  if (beq.length !== mEq) {
    throw new Error('Aeq and beq must have the same length. Aeq.length = ' + mEq + ', beq.length = ' + beq.length);
  }

  const AineqT = transpose(Aineq);
  const AeqT = transpose(Aeq);

  // Define the function for evaluating the objective and constraints
  function evalFunc(x, s, y, z, mu) {
    const Hx = matrixTimesVector(H, x);
    const Aeqx = matrixTimesVector(Aeq, x);
    const Aineqx = matrixTimesVector(Aineq, x);

    // Objective
    const f = 0.5 * dot(x, Hx) + dot(c, x); // 0.5 x' H x + c' x

    // Residuals
    let rGrad = add(Hx, c); // Hx + c + Aeq' y - Aineq' z
    if (mEq > 0 ) {
      const AeqTy = matrixTimesVector(AeqT, y);
      rGrad = add(rGrad, AeqTy);
    }
    if (mIneq > 0 ) {
      const AineqTz = matrixTimesVector(AineqT, z);
      rGrad = subtract(rGrad, AineqTz);
    }
    const rEq = subtract(Aeqx, beq); // Aeq x - beq
    const rIneq = subtract(subtract(Aineqx, s), bineq); // Aineq x - s - bineq
    const rS = subtract(elementwiseProduct(s, z), new Array(mIneq).fill(mu)); // SZe - mu e

    return { f, rGrad, rEq, rIneq, rS };
  }


  // Construct the augmented KKT system
  /*  [ H       Aeq'   Aineq' ]
   *  [ Aeq      0      0     ]
   *  [ Aineq    0   -Z^-1 S  ]
  */
  const m = n + mEq + mIneq;
  const KKT = zeroMatrix(m, m);
  setSubmatrix(KKT, H, 0, 0);
  setSubmatrix(KKT, AeqT, 0, n);
  setSubmatrix(KKT, AineqT, 0, n + mEq);
  setSubmatrix(KKT, Aeq, n, 0);
  setSubmatrix(KKT, Aineq, n + mEq, 0);
  function updateMatrix(s, z) {
    const minusZinvS = negate(elementwiseDivision(s, z));
    setSubdiagonal(KKT, minusZinvS, n + mEq, n + mEq);
  }

  // Define the function for computing the search direction
  function computeSearchDirection(s, z, L, ipiv, rGrad, rEq, rIneq, rS) {
    const rIneqMinusYinvrS = add(rIneq, elementwiseDivision(rS, z)); // Aineq x - s - bineq + Z^-1 (SZe - mue)
    const rhs = negate(rGrad.concat(rEq).concat(rIneqMinusYinvrS));

    // Solve the KKT system
    const d = solveUsingFactorization(L, ipiv, rhs);

    // Extract the search direction components
    const dx = d.slice(0, n);
    const dy = d.slice(n, n + mEq);
    const dz = negate(d.slice(n + mEq, n + mEq + mIneq));
    const ds = negate(elementwiseDivision(add(rS, elementwiseProduct(s, dz)), z)); // -Z^-1 (rS + S dz)

    return { dx, ds, dy, dz };
  }

  // Define the function for computing the step size
  function getMaxStep(v, dv) {
    return v.reduce((m, value, index) => dv[index] < 0 ? Math.min(-value/dv[index], m) : m, 1.0);
  }

  // Initialize primal and dual variables
  const x = new Array(n).fill(1.0);        // Primal variables
  const s = new Array(mIneq).fill(1.0);    // Slack variables for inequality constraints
  const y = new Array(mEq).fill(1.0);    // Multipliers for equality constraints
  const z = new Array(mIneq).fill(1.0); // Multipliers for inequality constraints
  
  function getMu(s, z) {
    return mIneq > 0 ? dot(s, z) / mIneq : 0;
  }

  function getResidualAndGap(s, z, rGrad, rEq, rIneq) {
    const res = norm(rGrad.concat(rEq).concat(rIneq));
    const gap = getMu(s, z);
    return { res, gap };
  }

  // Perform the interior point optimization
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const { f, rGrad, rEq, rIneq, rS } = evalFunc(x, s, y, z, 0);

    // Check the convergence criterion
    const { res, gap } = getResidualAndGap(s, z, rGrad, rEq, rIneq);
    console.log(`${iter}. f: ${f}, res: ${res}, gap: ${gap}`)
    if (res <= tol && gap <= tol) {
      break;
    }

    // Update and factorize KKT matrix
    updateMatrix(s, z);
    const [L, ipiv] = symmetricIndefiniteFactorization(KKT);

    // Use the predictor-corrector method

    // Compute affine scaling step
    const { dx : dxAff, ds : dsAff, dy : dyAff, dz : dzAff } = computeSearchDirection(s, z, L, ipiv, rGrad, rEq, rIneq, rS);
    const alphaAffP = getMaxStep(s, dsAff);
    const alphaAffD = getMaxStep(z, dzAff);
    const zAff = Array.from(z);
    const sAff = Array.from(s);
    vectorPlusEqScalarTimesVector(sAff, alphaAffP, dsAff);
    vectorPlusEqScalarTimesVector(zAff, alphaAffD, dzAff);
    const muAff = getMu(zAff, sAff);

    // Compute aggregated centering-corrector direction
    const mu = getMu(s, z);
    const sigma = mu > 0 ? Math.pow(muAff / mu, 3.0) : 0;
    const { rS : rSCenter } = evalFunc(x, s, y, z, sigma * mu);
    const rSCenterCorr = add(elementwiseProduct(dzAff, dsAff), rS);
    const { dx, ds, dy, dz } = computeSearchDirection(s, z, L, ipiv, rGrad, rEq, rIneq, rSCenterCorr);
    const alphaP = getMaxStep(s, ds);
    const alphaD = getMaxStep(z, dz);

    // Update the variables
    const fractionToBoundary = 0.995;
    vectorPlusEqScalarTimesVector(x, fractionToBoundary * alphaP, dx);
    vectorPlusEqScalarTimesVector(s, fractionToBoundary * alphaP, ds);
    vectorPlusEqScalarTimesVector(y, fractionToBoundary * alphaD, dy);
    vectorPlusEqScalarTimesVector(z, fractionToBoundary * alphaD, dz);
  }

  // Return the solution and objective value
  const { f, rGrad, rEq, rIneq, rS } = evalFunc(x, s, y, z, 0);
  const { res, gap } = getResidualAndGap(s, z, rGrad, rEq, rIneq);
  return { x, f, res, gap, iter };
}

// Helper functions for linear algebra operations
function filledVector(n, v) {
  return new Array(n).fill(v);
}

function zeroVector(n) {
  return filledVector(n, 0.0);
}

function filledMatrix(m, n, v) {
  return new Array(m).fill().map(() => new Array(n).fill(v));
}

function zeroMatrix(m, n) {
  return filledMatrix(m, n, 0.0);
}

function setSubmatrix(M, X, startI, startJ) {
  const m = X.length;
  if (M.length < m + startI) {
    throw new Error('Invalid submatrix row');
  }
  for (let i = 0; i < m; i++) {
    const si = i + startI;
    const n = X[i].length;
    if (M[si].length < n + startJ) {
      throw new Error('Invalid submatrix column');
    }
    for (let j = 0; j < n; j++) {
      M[si][j + startJ] = X[i][j];
    }
  }
}

function setSubdiagonal(M, d, startI, startJ) {
  const m = d.length;
  if (M.length < m + startI) {
    throw new Error('Invalid submatrix row');
  }
  for (let i = 0; i < m; i++) {
    M[i + startI][i + startJ] = d[i];
  }
}

function isVector(x) {
  return Array.isArray(x) && x.every(xi => typeof xi === 'number');
}

function assertIsVector(x, name) {
  if (!isVector(x)) {
    throw new Error('Invalid input type: ' + name + ' must be an array. ' + name + ': ' + x);
  }
}

function assertIsMatrix(A) {
  if (!Array.isArray(A) || A.some(row => !isVector(row))) {
    throw new Error('Invalid input type: A must be a matrix. A: ' + A);
  }
}

function diag(x) {
  assertIsVector(x, 'x');
  m = x.length;
  X = zeroMatrix(m, m);
  for (let i = 0; i < m; i++) {
    X[i][i] = x[i];
  }
  return X;
}

function assertAreEqualLengthVectors(x, y) {
  assertIsVector(x, 'x');
  assertIsVector(y, 'y');

  if (x.length !== y.length) {
    throw new Error('Invalid input shape: x and y must have the same length. x.length = ' + x.length + ', y.length = ' + y.length);
  }
}

function transpose(A) {
  assertIsMatrix(A);
  const m = A.length;
  const n = m > 0 ? A[0].length : 0;
  const B = zeroMatrix(n, m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      B[j][i] = A[i][j];
    }
  }
  return B;
}

function negate(x) {
  assertIsVector(x, 'x');
  return x.map(value => -value);
}

function elementwiseProduct(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value * y[index]);
}

function elementwiseDivision(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value / y[index]);
}

function vectorPlusEqScalarTimesVector(x, s, y) {
  assertAreEqualLengthVectors(x, y);
  for (let i = 0; i < x.length; i++) {
    x[i] += s * y[i];
  }
}

function matrixTimesVector(A, x) {
  assertIsMatrix(A);
  A.every(row => assertAreEqualLengthVectors(row, x));
  return A.map(ai => dot(ai, x));
}

function add(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value + y[index]);
}

function addVectors(...vectors) {
  return vectors.reduce((acc, vec) => add(acc, vec));
}

function subtract(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value - y[index]);
}

function subtractVectors(...vectors) {
  return vectors.reduce((acc, vec) => subtract(acc, vec));
}

function norm(x) {
  return Math.sqrt(dot(x, x));
}

function dot(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.reduce((sum, value, index) => sum + value * y[index], 0);
}

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
        for (let j = k; j < n; j++) {
          if (j != imax) {
            const v = Math.abs(A[imax][j]);
            if (v > rowmax) {
              rowmax = v;
              jmax = j;
            }
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
        for (let i = k + 1; i < n; i++) {
          for (let j = i; j < n; j++) {
            A[i][j] -= r1 * row[i] * row[j]
            A[j][i] = A[i][j];
          }
        }

        for (let i = k + 1; i < n; i++) {
          A[k][i] *= r1;
          A[i][k] *= r1;
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
            A[k][j] = wk;
            A[k + 1][j] = wkp1;
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
  let k = 0;
  while (k < n) {
    if (ipiv[k] >= 0) {
      // 1 x 1 diagonal block, interchange rows k and ipiv(k).
      const kp = ipiv[k];
      if (kp != k) {
        [b[k], b[kp]] = [b[kp], b[k]];
      }
      // Multiply by inv(L(k)), where L(k) is the transformation stored in column k of L.

      // Subroutine dger
      const temp = -b[k];
      for (let i = k + 1; i < n; i++) {
        b[i] += L[i][k] * temp;
      }
      
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
        // Subroutine dger
        const temp = -b[k];
        for (let i = k + 2; i < n; i++) {
          b[i] += L[i][k] * temp;
        }

        // Subroutine dger
        const temp2 = -b[k + 1];
        for (let i = k + 2; i < n; i++) {
          b[i] += L[i][k + 1] * temp2;
        }
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
  k = n - 1;
  while (k >= 0) {
    if (ipiv[k] >= 0) {
      // 1 x 1 diagonal block, multiply by inv(L**T(k)), where L(k) is the transformation stored in column k of L.

      if (k < n - 1) {
        // Subroutine dgemv 'Transpose' with alpha = -1 and beta = 1
        let temp = 0.0;
        for (let i = k + 1; i < n; ++i) {
          temp += L[i][k] * b[i];
        }
        b[k] -= temp;
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
        // Subroutine dgemv 'Transpose' with alpha = -1 and beta = 1
        let temp = 0.0;
        for (let i = k + 1; i < n; ++i) {
          temp += L[i][k] * b[i];
        }
        b[k] -= temp;

        let temp2 = 0.0;
        for (let i = k + 1; i < n; ++i) {
          temp2 += L[i][k - 1] * b[i];
        }
        b[k - 1] -= temp2;
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

// Parsing of objectives and constraints

function parseTerms(str) {
  const pattern = /([+-]?\s*\d*\.?\d*)\s*(?:\*)?\s*(\w+)(?:\^(\d+))?/g;
  let matches = [];
  while ((match = pattern.exec(str)) !== null) {
    let [_, coefficient, name, power] = match;
    power = parseInt(power || '1');
    sign = coefficient.includes('-') ? -1 : 1;
    coefficient = coefficient.replace('+', '').replace('-', '').replace(/\s/g, '');
    coefficient = parseFloat(coefficient || '1');
    if (!(name && coefficient && power)) {
      throw new Error('Error parsing string: ' + match);
    }
    matches.push({
      variable: name,
      coefficient: sign * coefficient,
      power: power
    });
  }
  return matches;
}

function validateVariables(allVariables, newVariables) {
  for (const key of newVariables) {
    if (!allVariables.includes(key)) {
      throw new Error('Variable "' + key + '" has no quadratic coefficient.');
    }
  }
}

function validatePowers(terms, allowedPowers) {
  for (const term of terms) {
    if (!allowedPowers.includes(term.power)) {
      throw new Error('Disallowed power: ' + term.power + '. Allowed powers: ' + allowedPowers);
    }
  }
}

function addOrInsert(dictionary, term) {
  dictionary[term.variable] = (term.variable in dictionary)
    ? dictionary[term.variable] + term.coefficient
    : term.coefficient;
}

function parseObjective(str) {
  const terms = parseTerms(str);
  let linearCoefficients = {};
  let quadraticCoefficients = {};
  validatePowers(terms, [1, 2]);
  for (const term of terms) {
    addOrInsert(term.power === 1 ? linearCoefficients : quadraticCoefficients, term);
  }
  for (const key in quadraticCoefficients) {
    if (quadraticCoefficients[key] <= 0) {
      throw new Error('Variable "' + key + '" has negative quadratic coefficient.');
    }
  }
  const variables = Object.keys(quadraticCoefficients).sort();
  const m = variables.length;

  const Q = zeroMatrix(m, m);
  for (let i = 0; i < m; ++i) {
    Q[i][i] = 2 * quadraticCoefficients[variables[i]];
  }

  const c = zeroVector(m);
  const linearKeys = Object.keys(linearCoefficients);
  validateVariables(variables, linearKeys);
  for (const key of linearKeys) {
    const index = variables.indexOf(key);
    c[index] = linearCoefficients[key];
  }

  return { Q, c, variables }
}

function parseConstraint(str) {
  const separator = str.includes('<=') ? '<=' : (str.includes('>=') ? '>=' : '=');
  const substrings = str.split(separator);
  if (substrings.length !== 2) {
    throw new Error('Error in parsing constraint: ' + str + ', substrings: ' + substrings);
  }
  const terms = parseTerms(substrings[0]);
  const rhs = parseFloat(substrings[1]);
  validatePowers(terms, [1]);
  let coefficients = {}
  for (const term of terms) {
    addOrInsert(coefficients, term);
  }
  return { coefficients, separator, rhs };
}

function parseConstraints(variables, constraints) {
  const m = variables.length;
  let ineqs = [];
  let eqs = [];
  for (const constraint of constraints) {
    const { coefficients, separator, rhs } = parseConstraint(constraint);
    if (separator === '=') {
      eqs.push({ coefficients, separator, rhs });
    }
    else {
      ineqs.push({ coefficients, separator, rhs });
    }
  }

  function createConstraints(cs) {
    const A = zeroMatrix(cs.length, m);
    const b = zeroVector(cs.length);
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      const sign = c.separator === '<=' ? -1 : 1;
      const constraintVariables = Object.keys(c.coefficients);
      validateVariables(variables, constraintVariables);
      for (const v in c.coefficients) {
        A[i][variables.indexOf(v)] = sign * c.coefficients[v];
      }
      b[i] = sign * c.rhs;
    }
    return { A, b };
  }

  const { A: Aeq, b: beq } = createConstraints(eqs);
  const { A: Aineq, b: bineq } = createConstraints(ineqs);

  return { Aeq, beq, Aineq, bineq };
}

function solveQP(Q, c, Aeq, beq, Aineq, bineq, variables = []) {
  let solutionElement = document.getElementById("solution");
  
  try {
    const start = performance.now();
    const {x, f, res, gap, iter} = interiorPointQP(Q, c, Aeq, beq, Aineq, bineq);
    const end = performance.now();

    let tableStr = '<table>';
    function addRow(str, val) {
      tableStr += `<tr><td>${str}</td><td>${val}</td></tr>`;
    }

    addRow('Objective value', f);
    addRow('Number of iterations', iter);
    addRow('Residual', res);
    addRow('Gap', gap);
    addRow('Elapsed time', `${end - start} milliseconds`);
    for (let i = 0; i < x.length; i++) {
      addRow(variables.length === x.length ? variables[i] : `x${i}`, x[i]);
    }
    addRow('Variable vector', x);
    tableStr += '</table>';

    solutionElement.innerHTML = tableStr;

  } catch (error) {
    solutionElement.innerHTML = `Error ${error.lineNumber}: ${error.message}`;
  }
}

function test() {
  const { Q, c, variables } = parseObjective('-5 x + 7*y + x^2 + 13x1^2 + 14x2 -   15*x3 + x3^2 + a^2 + 2y^2 + 0.33 x2^2');
  const { Aeq, beq, Aineq, bineq } = parseConstraints(variables, ['x + 2 y = 3', '2x + 3*y <= 5', 'x1 - 0.3x3 >= 3']);
  console.log('variables: ' + variables);
  console.log('Q: ' + Q);
  console.log('c: ' + c);
  console.log('Aeq: ' + Aeq);
  console.log('beq: ' + beq);
  console.log('Aineq: ' + Aineq);
  console.log('bineq: ' + bineq);
  solveQP(Q, c, Aeq, beq, Aineq, bineq, variables);
}

function solveTestProblem() {
  if (false) {
    let n = 100;
    const Q = zeroMatrix(n, n);
    const c = zeroVector(n);
    const Aineq = zeroMatrix(n, n);
    const bineq = zeroVector(n);
    for (let i = 0; i < n; ++i) {
      Q[i][n-i-1] = 1.2 + 0.1;
      Q[n-i-1][i] = 1.2 + 0.1;

      Q[i][i] = i + 3;
      c[i] = -0.5 * i;
      Aineq[i][i] = 1; // x[i] >= i
      bineq[i] = i * i * 0.01;
    }
    let Aeq = zeroMatrix(0, 0);
    let beq = zeroVector(0);
    Aeq = zeroMatrix(1, n);
    beq = zeroVector(1);
    Aeq[0][1] = 1; // x[0] - 2 x[1] = 0
    Aeq[0][2] = -2;
    solveQP(Q, c, Aeq, beq, Aineq, bineq);
  }
  else {
    let n = 2;
    const Q = zeroMatrix(n, n);
    const c = zeroVector(n);
    let t0 = 1300;
    let t1 = 50;
    let a00 = 809;
    let a01 = 359;
    let a10 = 25;
    let a11 = 77;


    // e' x = 1
    const Aeq = filledMatrix(1, n, 1.0);
    const beq = filledVector(1, 1.0);

    // x >= 0
    const Aineq = diag(filledVector(n, 1.0));
    const bineq = zeroVector(n);

    let k = 0;
    Q[0][0] = (a00**2) / t0**2 + k;
    Q[1][1] = (a01**2) / t0**2 + k;
    Q[0][1] = (a00 * a01) / t0**2;
    Q[1][0] = (a00 * a01) / t0**2;
    c[0] = -t0 * a00 / t0**2;
    c[1] = -t0 * a01 / t0**2;
    console.log(Q)

    const mIneq = Aineq.length;
    const mEq = Aeq.length;

    const x = new Array(n).fill(1.0);        // Primal variables
    const s = new Array(mIneq).fill(1.0);    // Slack variables for inequality constraints
    const y = new Array(mEq).fill(1.0);    // Multipliers for equality constraints
    const z = new Array(mIneq).fill(1.0); // Multipliers for inequality constraints

    const AineqT = transpose(Aineq);
    const AeqT = transpose(Aeq);

    const m = n + mEq + mIneq;
    const KKT = zeroMatrix(m, m);
    setSubmatrix(KKT, Q, 0, 0);
    setSubmatrix(KKT, AeqT, 0, n);
    setSubmatrix(KKT, AineqT, 0, n + mEq);
    setSubmatrix(KKT, Aeq, n, 0);
    setSubmatrix(KKT, Aineq, n + mEq, 0);
    const minusZinvS = negate(elementwiseDivision(s, z));
    setSubdiagonal(KKT, minusZinvS, n + mEq, n + mEq);

    console.log('kkt')
    console.log(KKT)

    v = [1,2, 3, 4, 5];

    {
      console.log('old:');
      const [L, D] = symmetricIndefiniteFactorization_unstable(KKT);
      console.log(L)
      console.log(D)
      const b = solveUsingFactorization_unstable(L, D, v);
      console.log('b: ' + b);
    }

    {
      console.log('new:');
      const [L, P] = symmetricIndefiniteFactorization(KKT);
      console.log(L);
      console.log(P);
      const b = solveUsingFactorization(L, P, v);
      console.log('b: ' + b);
    }

    console.log('Solve QP');
    solveQP(Q, c, Aeq, beq, Aineq, bineq);
  }
}

// Functions relating to buttons on the html page

function solve() {
  const objective = document.getElementById("objective").value;
  const table = document.getElementById("optimization-problem");
  let constraints = [];
  for (let i = 2; i < table.rows.length; i++) {
    const constraint = document.getElementById(`constraint-${i}`).textContent;
    constraints.push(constraint);
  }
  try {
    const { Q, c, variables } = parseObjective(objective);
    const { Aeq, beq, Aineq, bineq } = parseConstraints(variables, constraints)
    solveQP(Q, c, Aeq, beq, Aineq, bineq, variables);
  } catch (error) {
    let solutionElement = document.getElementById("solution");
    solutionElement.innerHTML = `Error ${error.lineNumber}: ${error.message}`;
  }
}

function clear() {
  const table = document.getElementById("optimization-problem");
  for (let i = 2; i < table.rows.length; i++) {
    table.deleteRow(i);
  }
  solutionElement = document.getElementById("solution");
  solutionElement.innerHTML = '';
}

function addConstraint() {
  const table = document.getElementById("optimization-problem");
  const m = table.rows.length;
  const row = table.insertRow(m);
  const constraint = document.getElementById("constraint");
  row.innerHTML = `<td></td><td id="constraint-${m}">` + constraint.value + '</td>';
  constraint.value = '';
}

document.getElementById("clear").addEventListener("click", clear);
document.getElementById("solve").addEventListener("click", solve);
document.getElementById("test").addEventListener("click", solveTestProblem);
document.getElementById("add-constraint").addEventListener("click", addConstraint);
