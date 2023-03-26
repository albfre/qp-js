function interiorPointQP(H, c, Aineq, bineq, Aeq, beq, tol=1e-8, maxIter=1000) {
  /* minimize 0.5 x' H x + c' x
   *   st     Aineq x <= bineq
   *          Aeq x = beq
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
    const AineqTY = matrixTimesVector(AineqT, y);
    const AeqTZ = matrixTimesVector(AeqT, z);

    // Objective
    const f = 0.5 * dot(x, Hx) + dot(c, x); // 0.5 x' H x + c' x

    // Residuals
    let rGrad = add(Hx, c); // Hx + Aineq' y + Aeq' z + c
    if (mIneq > 0 ) {
      rGrad = add(rGrad, AineqTY);
    }
    if (mEq > 0 ) {
      rGrad = add(rGrad, AeqTZ);
    }
    const rIneq = subtract(add(matrixTimesVector(Aineq, x), s), bineq); // Aineq x + s - bineq
    const rEq = subtract(matrixTimesVector(Aeq, x), beq); // Aeq x - beq
    const rS = subtract(elementwiseVectorProduct(s, y), new Array(mIneq).fill(mu)); // SYe - mu e

    return { f, rGrad, rIneq, rEq, rS };
  }

  // Construct the augmented KKT system
  /*  [ H       Aineq'     Aeq']
   *  [ Aineq  -Y^-1 S      0  ]
   *  [ Aeq       0         0  ]
  */
  const m = n + mIneq + mEq;
  const KKT = zeroMatrix(m, m);
  setSubmatrix(KKT, H, 0, 0);
  setSubmatrix(KKT, AineqT, 0, n);
  setSubmatrix(KKT, AeqT, 0, n + mIneq);
  setSubmatrix(KKT, Aineq, n, 0);
  setSubmatrix(KKT, Aeq, n + mIneq, 0);

  // Define the function for computing the search direction
  function computeSearchDirection(x, s, y, z, mu) {
    const minusYinvS = negate(elementwiseVectorDivision(s, y));
    setSubdiagonal(KKT, minusYinvS, n, n);
  
    const { f, rGrad, rIneq, rEq, rS } = evalFunc(x, s, y, z, mu);
    const rIneqMinusYinvrS = subtract(rIneq, elementwiseVectorDivision(rS, y)); // Aineq x + s - bineq - Y^-1 (SYe - mue)
    const rhs = negate(rGrad.concat(rIneqMinusYinvrS).concat(rEq));

    // Solve the KKT system
    const d = solveSymmetricIndefinite(KKT, rhs);

    // Extract the search direction components
    const dx = d.slice(0, n);
    const dy = d.slice(n, n + mIneq);
    const dz = d.slice(n + mIneq, n + mIneq + mEq);
    const ds = negate(elementwiseVectorDivision(add(rS, elementwiseVectorProduct(s, dy)), y)); // -Y^-1 (rS + S dy)

    return { dx, ds, dy, dz, ds };
  }

  // Define the function for computing the step size
  function computeStepSize(x, s, y, z, dx, ds, dy, dz) {
    const alpha = 0.995;

    function getMaxStep(v, dv) {
      let n = v.length;
      let maxStep = 1.0;
      for (let i = 0; i < n; i++) {
        if (dv[i] < 0) {
          maxStep = Math.min(maxStep, -v[i] / dv[i]); // v + alpha dv > 0 => alpha > -v/dv
        }
      }
      return maxStep;
    }

    // Compute the maximum step size for y and s
    const maxYStep = getMaxStep(y, dy);
    const maxSStep = getMaxStep(s, ds);

    // Compute the step size
    return alpha * Math.min(maxYStep, maxSStep);
  }

  // Initialize primal and dual variables
  const x = new Array(n).fill(1.0);        // Primal variables
  const s = new Array(mIneq).fill(1.0);    // Slack variables for inequality constraints
  const y = new Array(mIneq).fill(1.0);    // Multipliers for inequality constraints
  const z = new Array(mEq).fill(1.0); // Multipliers for equality constraints

  // Initialize barrier parameter
  const sigma = 0.8;
  let mu = dot(s, y) / mIneq;

  // Perform the interior point optimization
  let iter = 0;
  while (iter < maxIter) {
    mu = (mIneq > 0 ? dot(s, y) / mIneq : mu) * sigma;
    iter++;

    // Compute the objective and residuals
    const { f, rGrad, rIneq, rEq, rS } = evalFunc(x, s, y, z, mu);
    console.log('f: ' + f);

    // Check the convergence criterion
    const normRes = norm(rGrad.concat(rIneq).concat(rEq));
    const gap = mIneq > 0 ? dot(s, y) / mIneq : 0;
    console.log('res: ' + normRes + ', gap: ' + gap)
    if (normRes <= tol && gap <= tol) {
      break;
    }

    // Compute the search direction
    const { dx, ds, dy, dz } = computeSearchDirection(x, s, y, z, mu);

    // Compute the step size
    const stepSize = computeStepSize(x, s, y, z, dx, ds, dy, dz);

    // Update the variables
    vectorPlusEqScalarTimesVector(x, stepSize, dx);
    vectorPlusEqScalarTimesVector(s, stepSize, ds);
    vectorPlusEqScalarTimesVector(y, stepSize, dy);
    vectorPlusEqScalarTimesVector(z, stepSize, dz);
  }

  // Return the solution and objective value
  const { f, rGrad, rIneq, rEq, rS } = evalFunc(x, s, y, z, mu);
  return { x, f, iter };
}


// Helper functions for linear algebra operations
function zeroVector(n) {
  return new Array(n).fill(0.0);
}

function zeroMatrix(m, n) {
  return new Array(m).fill().map(() => new Array(n).fill(0.0));
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
  return Array.isArray(x) && x.every(xi => typeof xi == 'number');
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
    throw new Error('Invalid input shape: x and y must have the same length. x.length = ' + x.length + ", y.length = " + y.length);
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

function elementwiseVectorProduct(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value * y[index]);
}

function elementwiseVectorDivision(x, y) {
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

function norm(x) {
  return Math.sqrt(dot(x, x));
}

function dot(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.reduce((sum, value, index) => sum + value * y[index], 0);
}

function chol(A) {
  const n = A.length;
  const L = zeroMatrix(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A[i][j];
      for (let k = 0; k < j; k++) {
        s -= L[i][k] * L[j][k];
      }
      if (i == j) {
        L[i][j] = Math.sqrt(s);
      } else {
        L[i][j] = s / L[j][j];
      }
    }
  }
  return L;
}

function solveChol(L, b) {
  const n = L.length;
  const y = zeroVector(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let j = 0; j < i; j++) {
      s -= L[i][j] * y[j];
    }
    y[i] = s / L[i][i];
  }
  const x = zeroVector(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = y[i];
    for (let j = i + 1; j < n; j++) {
      s -= L[j][i] * x[j];
    }
    x[i] = s / L[i][i];
  }
  return x;
}

function luFactorization(A) {
  const n = A.length;
  const L = new Array(n).fill(null).map(() => new Array(n).fill(0));
  const U = new Array(n).fill(null).map(() => new Array(n).fill(0));

  // Perform Gaussian elimination with partial pivoting
  for (let j = 0; j < n; j++) {
    // Compute the jth column of U
    for (let i = 0; i <= j; i++) {
      let sum = 0;
      for (let k = 0; k < i; k++) {
        sum += L[i][k] * U[k][j];
      }
      U[i][j] = A[i][j] - sum;
    }

    // Compute the jth column of L
    for (let i = j + 1; i < n; i++) {
      let sum = 0;
      for (let k = 0; k < j; k++) {
        sum += L[i][k] * U[k][j];
      }
      L[i][j] = (A[i][j] - sum) / U[j][j];
    }
  }

  // Set the diagonal elements of L to 1
  for (let i = 0; i < n; i++) {
    L[i][i] = 1;
  }

  return [L, U];
}

function luSolve(A, b) {
  // Perform LU factorization of A
  const [L, U] = luFactorization(A);

  // Solve Ly = b using forward substitution
  const n = L.length;
  const y = new Array(n);
  y[0] = b[0] / L[0][0];
  for (let i = 1; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = (b[i] - sum) / L[i][i];
  }

  // Solve Ux = y using backward substitution
  const x = new Array(n);
  x[n - 1] = y[n - 1] / U[n - 1][n - 1];
  for (let i = n - 2; i >= 0; i--) {
    let sum = 0;
    for (let j = i + 1; j < n; j++) {
      sum += U[i][j] * x[j];
    }
    x[i] = (y[i] - sum) / U[i][i];
  }

  return x;
}


/**
 * Computes the symmetric indefinite factorization of a matrix A using LDL^T decomposition
 * @param {Array<Array<number>>} A - The input matrix
 * @returns {Array<Array<number>>} - An array containing the factorization: [L,D,p]
 */
function symmetricIndefiniteFactorization(A) {
  const n = A.length;
  const L = Array.from(Array(n), () => new Array(n).fill(0));
  const D = new Array(n).fill(0);
  const p = new Array(n);

  for (let i = 0; i < n; i++) {
    p[i] = i;

    // Compute the (i,i) entry of D
    let d_ii = A[i][i];
    for (let k = 0; k < i; k++) {
      d_ii -= L[i][k] ** 2 * D[k];
    }

    // Check for singularity
    if (d_ii === 0) {
      throw new Error("Matrix is singular");
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

  return [L, D, p];
}

/**
 * Solves a system of linear equations Ax = b using the symmetric indefinite factorization of A
 * @param {Array<Array<number>>} L - The lower-triangular matrix from the factorization
 * @param {Array<number>} D - The diagonal matrix from the factorization
 * @param {Array<number>} p - The permutation vector from the factorization
 * @param {Array<number>} b - The right-hand side vector
 * @returns {Array<number>} - The solution vector x
 */
function solveSymmetricIndefiniteUsingFactorization(L, D, p, b) {
  const n = L.length;
  const x = new Array(n).fill(0);
  const y = new Array(n).fill(0);

  // Forward substitution: solve Ly = Pb
  for (let i = 0; i < n; i++) {
    let sum = 0;
    for (let j = 0; j < i; j++) {
      sum += L[i][j] * y[j];
    }
    y[i] = b[p[i]] - sum;
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

function solveSymmetricIndefinite(A, b) {
  [L, D, p] = symmetricIndefiniteFactorization(A);
  return solveSymmetricIndefiniteUsingFactorization(L, D, p, b);
}

/*
    // Define the function to parse the user input and call the solver

    function solve() {
      // Parse the objective function
      const objectiveStr = document.getElementById("objective").value.trim();
      const objectiveCoefficients = objectiveStr.split(/ +/).map(parseFloat);
      const n = objectiveCoefficients.length;
      const P = zeroMatrix(n, n);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          P[i][j] = objectiveCoefficients[i] * objectiveCoefficients[j];
        }
      }
      // Parse the inequality constraints
      const inequalityTable = document.getElementById("inequalities");
      const m1 = inequalityTable.rows.length;
      const A1 = zeroMatrix(m1, n);
      const b1 = zeroVector(m1);
      for (let i = 0; i < m1; i++) {
        const leq = inequalityTable.rows[i].querySelector('input[name="inequality-type-' + i + '"]:checked').value === "leq";
        const coefficients = Array.from(inequalityTable.rows[i].querySelectorAll('input[type="number"]')).map(parseFloat);
        for (let j = 0; j < n; j++) {
          A1[i][j] = coefficients[j];
        }
        if (leq) {
          b1[i] = coefficients[n];
        } else {
          for (let j = 0; j < n; j++) {
            A1[i][j] *= -1;
          }
          b1[i] = -coefficients[n];
        }
      }

      // Parse the equality constraints
      const equalityTable = document.getElementById("equalities");
      const m2 = equalityTable.rows.length;
      const A2 = zeroMatrix(m2, n);
      const b2 = zeroVector(m2);
      for (let i = 0; i < m2; i++) {
        const coefficients = Array.from(equalityTable.rows[i].querySelectorAll('input[type="number"]')).map(parseFloat);
        for (let j = 0; j < n; j++) {
          A2[i][j] = coefficients[j];
        }
        b2[i] = coefficients[n];
      }
      // Call the solver
      const solution = interiorPointQP(P, zeroVector(n), A1, b1, A2, b2);
      document.getElementById("solution").innerHTML = solution.join(", ");
    }

    // Define the function to add a new inequality constraint
    function addInequality() {
      const inequalityTable = document.getElementById("inequalities");
      const m = inequalityTable.rows.length;
      const row = inequalityTable.insertRow(m);
      row.innerHTML = '<td><input type="radio" name="inequality-type-' + m + '" value="leq" checked></td>' + 
                      '<td><input type="number" name="inequality-' + m + '-1" value="0"></td>' + 
                      '<td><input type="number" name="inequality-' + m + '-2" value="0"></td>';
    }

    // Define the function to add a new equality constraint
    function addEquality() {
      const equalityTable = document.getElementById("equalities");
      const m = equalityTable.rows.length;
      const row = equalityTable.insertRow(m);
      row.innerHTML = '<td><input type="radio" name="equality-type-' + m + '" value="eq" checked></td>' + 
                      '<td><input type="number" name="equality-' + m + '-1" value="0"></td>' + 
                      '<td><input type="number" name="equality-' + m + '-2" value="0"></td>';
    }

    // Add event listeners to the buttons
    document.getElementById("add-inequality").addEventListener("click", addInequality);
    document.getElementById("add-equality").addEventListener("click", addEquality);
    */

function solve() {
  /*
  const Q = zeroMatrix(2, 2);
  Q[0][0] = 8;
  Q[0][1] = 2;
  Q[1][0] = 2;
  Q[1][1] = 4;
  const c = zeroVector(2);
  c[0] = -2;
  let Aineq = zeroMatrix(0,0);
  let bineq = zeroVector(0);
  if (true) {
    // x <= -1
    Aineq = zeroMatrix(1,2);
    Aineq[0][0] = 1;
    bineq = zeroVector(1);
    bineq[0] = -1;
  }

  let Aeq = zeroMatrix(0,0);
  let beq = zeroVector(0);
  if (true) {
    // x + y = 0
    Aeq = zeroMatrix(1,2);
    beq = zeroVector(1);
    Aeq[0][0] = 1;
    Aeq[0][1] = 1;
  }
  */
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
    Aineq[i][i] = -1; // -x[i] <= -i => x[i] >= i
    bineq[i] = -i * i * 0.01;
  }
  let Aeq = zeroMatrix(0, 0);
  let beq = zeroVector(0);
  /*
  const Aeq = zeroMatrix(3, n);
  const beq = zeroVector(3);
  Aeq[0][1] = 1; // x[0] + x[1] = 0
  Aeq[0][2] = 1;
  Aeq[1][3] = 1; // x[0] + x[1] = 0
  Aeq[1][4] = 1;
  Aeq[2][5] = 1; // x[0] + x[1] = 0
  Aeq[2][6] = 1;
  */

  solutionElement = document.getElementById("solution");
  
  try {
    const start = performance.now();
    const {x, f, iter} = interiorPointQP(Q, c, Aineq, bineq, Aeq, beq);
    const end = performance.now();
    console.log(`Elapsed time: ${end - start} milliseconds`);

    solutionElement.innerHTML = 'x: ' + x + ', obj: ' + f + ', iters: ' + iter;
  } catch (error) {
    solutionElement.innerHTML = `Error: ${error.message}`;
  }
}

document.getElementById("solve").addEventListener("click", solve);
