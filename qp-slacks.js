const useLDLT = true;

function interiorPointQP_old(H, c, Aeq, beq, Aineq, bineq, tol=1e-8, maxIter=100) {
  /* minimize 0.5 x' H x + c' x
   *   st    Aeq x = beq
   *         Aineq x >= bineq
   */
   const n = H.length;
   const mEq = Aeq.length;
   const mIneq = Aineq.length;
   const A = Aeq.concat(Aineq);
   let lA = createVector(mEq + mIneq);
   let uA = createVector(mEq + mIneq);
   for (let i = 0; i < mEq; ++i) {
     lA[i] = beq[i];
     uA[i] = beq[i];
   }
   for (let i = 0; i < mIneq; ++i) {
     lA[i + mEq] = bineq[i];
     uA[i + mEq] = null;
   }
   let lx = createVector(n).fill(null);
   let ux = createVector(n).fill(null);
   return interiorPointQP(H, c, A, lA, uA, lx, ux, tol, maxIter);
}

function interiorPointQP(H, c, A, lA, uA, lx, ux, tol=1e-8, maxIter=100) {
  /* minimize 0.5 x' H x + c' x
   *   st    lA <= A <= uA
   *         lx <= x <= ux 
   */

  // Matrix sizes
  const n = H.length;
  const m = A.length;

  // Preconditions
  sanityCheck(H.every(row => row.length === n), 'H is not a square matrix');
  sanityCheck(A.every(row => row.length === n), 'All rows of A must have the same length as H');
  sanityCheck(lA.length === m, 'A and lA must have the same length. A.length = ' + m + ', lA.length = ' + lA.length);
  sanityCheck(uA.length === m, 'A and uA must have the same length. A.length = ' + m + ', uA.length = ' + uA.length);
  sanityCheck(lx.length === n, 'H and lx must have the same length. H.length = ' + n + ', lx.length = ' + lx.length);
  sanityCheck(ux.length === n, 'H and ux must have the same length. H.length = ' + n + ', ux.length = ' + ux.length);
  sanityCheck(lA.every((value, index) => value !== null || uA[index] !== null), 'Either lA[i] or uA[i] must be defined');

  const AT = transpose(A);

  const evalSlackResiduals = (v, mu) => ({
    r_g: lA.map((lAi, i) => lAi === null ? 0.0 : v.g[i] * v.lambda_g[i] - mu), // G Lambda_g e - mu e
    r_t: uA.map((uAi, i) => uAi === null ? 0.0 : v.t[i] * v.lambda_t[i] - mu), // T Lambda_t e - mu e
    r_y: lx.map((lxi, i) => lxi === null ? 0.0 : v.y[i] * v.lambda_y[i] - mu), // Y Lambda_y e - mu e
    r_z: ux.map((uxi, i) => uxi === null ? 0.0 : v.z[i] * v.lambda_z[i] - mu) // Z Lambda_z e - mu e
  });

  // Define the function for evaluating the objective and constraints and compute the residuals
  const evalFunc = (v) => {
    const Hx = matrixTimesVector(H, v.x);
    const Ax = matrixTimesVector(A, v.x);

    // Objective value
    const f = 0.5 * dot(v.x, Hx) + dot(c, v.x); // 0.5 x' H x + c' x

    // Residuals
    const r = structuredClone(v);

    r.x = Hx.map((Hxi, i) => Hxi + c[i] - v.lambda_y[i] + v.lambda_z[i]); // Hx + c + A' lambda_A - lambda_y + lambda_z
    if (m > 0) {
      const ATlambda_A = matrixTimesVector(AT, v.lambda_A);
      r.x = r.x.map((rxi, i) => rxi + ATlambda_A[i]);
    }

    r.s = v.lambda_A.map((lambda_Ai, i) => -lambda_Ai - v.lambda_g[i] + v.lambda_t[i]); // -lambda_A - lambda_g + lambda_t
    ({ r_g : r.g, r_t : r.t, r_y : r.y, r_z : r.z } = evalSlackResiduals(v, 0.0));
    r.lambda_A = Ax.map((Axi, i) => Axi - v.s[i]); // Ax - s
    r.lambda_g = lA.map((lAi, i) => lAi === null ? 0.0 : v.s[i] - v.g[i] - lAi); // s - g - lA
    r.lambda_t = uA.map((uAi, i) => uAi === null ? 0.0 : v.s[i] + v.t[i] - uAi); // s + t - uA
    r.lambda_y = lx.map((lxi, i) => lxi === null ? 0.0 : v.x[i] - v.y[i] - lxi); // x - y - lx
    r.lambda_z = ux.map((uxi, i) => uxi === null ? 0.0 : v.x[i] + v.z[i] - uxi); // x + z - ux

    return { f, r };
  }

  // Construct the KKT system
  /*  [ H + Y^-1 Lambda_y + Z^-1 Lambda_z  A'                                  ]
   *  [ A                                  -(G^-1 Lambda_g + T^-1 Lambda_t)^-1 ]
  */
  const KKT = createMatrix(n + m, n + m);
  setSubmatrix(KKT, H, 0, 0);
  setSubmatrix(KKT, AT, 0, n);
  setSubmatrix(KKT, A, n, 0);
  const hDiag = H.map((row, i) => row[i]);

  const updateMatrix = (v) => {
    const newHDiag = hDiag.map((val, i) => val + v.lambda_y[i] / v.y[i] + v.lambda_z[i] / v.z[i]);
    setSubdiagonal(KKT, newHDiag, 0, 0);
    const d = Array.from({ length: m }, (_, i) => -1.0 / (v.lambda_g[i] / v.g[i] + v.lambda_t[i] / v.t[i]));
    setSubdiagonal(KKT, d, n, n);
  }

  // Define the function for computing the search direction
  const computeSearchDirection = (L, ipivOrD, v, r) => {
    const yPart = v.y.map((yi, i) => (v.lambda_y[i] * r.lambda_y[i] + r.y[i]) / yi); // Y^-1 (L_y r_lambda_y + r_y)
    const zPart = v.z.map((zi, i) => (v.lambda_z[i] * r.lambda_z[i] - r.z[i]) / zi); // Z^-1 (L_z r_lambda_z - r_z)
    const rhs1 = r.x.map((rxi, i) => -(rxi + yPart[i] + zPart[i]));
    
    const gtPart = v.g.map((gi, i) => v.lambda_g[i] / gi + v.lambda_t[i] / v.t[i]); // G^-1 L_g  + T^-1 L_t
    const gPart = v.g.map((gi, i) => (v.lambda_g[i] * r.lambda_g[i] + r.g[i]) / gi); // G^-1 (L_g r_lambda_g + r_g)
    const tPart = v.t.map((ti, i) => (v.lambda_t[i] * r.lambda_t[i] - r.t[i]) / ti); // T^-1 (L_t r_lambda_t - r_t)
    const sPart = r.s.map((rsi, i) => rsi + tPart[i] + gPart[i]);
    const rhs2 = r.lambda_A.map((rLambda_Ai, i) => -(rLambda_Ai + sPart[i] / gtPart[i]));

    const rhs = [...rhs1, ...rhs2];

    // Solve the KKT system
    const delta = useLDLT ? solveLDLT(L, ipivOrD, rhs) : solveUsingFactorization(L, ipivOrD, rhs);

    // Extract the search direction components
    const d = structuredClone(v);

    d.x = delta.slice(0, n);
    d.lambda_A = delta.slice(n, n + m);
    d.s = d.lambda_A.map((dLambda_Ai, i) => (dLambda_Ai - sPart[i]) / gtPart[i]);
    d.lambda_g = lA.map((lAi, i) => lAi === null ? 0.0 : -(v.lambda_g[i] * (d.s[i] + r.lambda_g[i]) + r.g[i]) / v.g[i]);
    d.lambda_t = uA.map((uAi, i) => uAi === null ? 0.0 : (v.lambda_t[i] * (d.s[i] + r.lambda_t[i]) - r.t[i]) / v.t[i]);
    d.lambda_y = lx.map((lxi, i) => lxi === null ? 0.0 : -(v.lambda_y[i] * (d.x[i] + r.lambda_y[i]) + r.y[i]) / v.y[i]);
    d.lambda_z = ux.map((uxi, i) => uxi === null ? 0.0 : (v.lambda_z[i] * (d.x[i] + r.lambda_z[i]) - r.z[i]) / v.z[i]);
    d.g = lA.map((lAi, i) => lAi === null ? 0.0 : -(v.g[i] * d.lambda_g[i] + r.g[i]) / v.lambda_g[i]);
    d.t = uA.map((uAi, i) => uAi === null ? 0.0 : -(v.t[i] * d.lambda_t[i] + r.t[i]) / v.lambda_t[i]);
    d.y = lx.map((lxi, i) => lxi === null ? 0.0 : -(v.y[i] * d.lambda_y[i] + r.y[i]) / v.lambda_y[i]);
    d.z = ux.map((uxi, i) => uxi === null ? 0.0 : -(v.z[i] * d.lambda_z[i] + r.z[i]) / v.lambda_z[i]);

    return d;
  }

  // Initialize primal and dual variables
  const variables = {
    x: createVector(n, 1.0),
    s: createVector(m, 1.0), // Slack variables for inequality constraints
    g: createVector(m, 1.0), // Slack for s
    t: createVector(m, 1.0), // Slack for s
    y: createVector(n, 1.0), // Slack for x
    z: createVector(n, 1.0), // Slack for x
    lambda_A: createVector(m, 1.0),
    lambda_g: lA.map(lAi => lAi === null ? 0.0 : 1.0),
    lambda_t: uA.map(uAi => uAi === null ? 0.0 : 1.0),
    lambda_y: lx.map(lxi => lxi === null ? 0.0 : 1.0),
    lambda_z: ux.map(uxi => uxi === null ? 0.0 : 1.0)
  };
  
  const getMu = (v) =>
    (m > 0 ? (dot(v.g, v.lambda_g) + dot(v.t, v.lambda_t)) / (2 * m) : 0.0) + (n > 0 ? (dot(v.y, v.lambda_y) + dot(v.z, v.lambda_z)) / (2 * n) : 0.0);

  const getResidualNorm = (r) =>
    norm([].concat(r.x, r.s, r.g, r.t, r.y, r.z, r.lambda_A, r.lambda_g, r.lambda_t, r.lambda_y, r.lambda_z));

  // Define the function for computing the step size
  const getMaxStep = (v, dv) => {
    const getMaxStepSingle = (vs, dvs) => vs.reduce((m, value, i) => dvs[i] < 0 ? Math.min(-value / dvs[i], m) : m, 1.0);
    const alphaG = getMaxStepSingle(v.g, dv.g);
    const alphaT = getMaxStepSingle(v.t, dv.t);
    const alphaY = getMaxStepSingle(v.y, dv.y);
    const alphaZ = getMaxStepSingle(v.z, dv.z);
    const alphaP = Math.min(alphaG, alphaT, alphaY, alphaZ);

    const alphaLambdaG = getMaxStepSingle(v.lambda_g, dv.lambda_g);
    const alphaLambdaT = getMaxStepSingle(v.lambda_t, dv.lambda_t);
    const alphaLambdaY = getMaxStepSingle(v.lambda_y, dv.lambda_y);
    const alphaLambdaZ = getMaxStepSingle(v.lambda_z, dv.lambda_z);
    const alphaD = Math.min(alphaLambdaG, alphaLambdaT, alphaLambdaY, alphaLambdaZ);
    
    return { alphaP, alphaD };
  };

  const updateVariables = (v, d, alphaP, alphaD) => {
    vectorPlusEqScalarTimesVector(v.x, alphaP, d.x);
    vectorPlusEqScalarTimesVector(v.s, alphaP, d.s);
    vectorPlusEqScalarTimesVector(v.g, alphaP, d.g);
    vectorPlusEqScalarTimesVector(v.t, alphaP, d.t);
    vectorPlusEqScalarTimesVector(v.y, alphaP, d.y);
    vectorPlusEqScalarTimesVector(v.z, alphaP, d.z);
    vectorPlusEqScalarTimesVector(v.lambda_A, alphaD, d.lambda_A);
    vectorPlusEqScalarTimesVector(v.lambda_g, alphaD, d.lambda_g);
    vectorPlusEqScalarTimesVector(v.lambda_t, alphaD, d.lambda_t);
    vectorPlusEqScalarTimesVector(v.lambda_y, alphaD, d.lambda_y);
    vectorPlusEqScalarTimesVector(v.lambda_z, alphaD, d.lambda_z);
  };

  // Perform the interior point optimization
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const { f, r : residuals } = evalFunc(variables);

    // Check the convergence criterion
    const residualNorm = getResidualNorm(residuals);
    const mu = getMu(variables);
    console.log(`${iter}. f: ${f}, res: ${residualNorm}, gap: ${mu}`)
    if (residualNorm <= tol && mu <= tol) {
      break;
    }

    // Update and factorize KKT matrix
    updateMatrix(variables);
    const [L, ipivOrD] = useLDLT ? ldltDecomposition(KKT) : symmetricIndefiniteFactorization(KKT);

    // Use the predictor-corrector method

    // Compute affine scaling step
    const delta_aff = computeSearchDirection(L, ipivOrD, variables, residuals);
    const { alphaP : alphaP_aff, alphaD : alphaD_aff } = getMaxStep(variables, delta_aff);

    const v_aff = structuredClone(variables);
    updateVariables(v_aff, delta_aff, alphaP_aff, alphaD_aff);
    const muAff = getMu(v_aff);

    // Compute aggregated centering-corrector direction
    const sigma = mu > 0 ? Math.pow(muAff / mu, 3.0) : 0;

    const r_center = evalSlackResiduals(variables, sigma * mu);
    residuals.g = lA.map((lAi, i) => lAi === null ? 0.0 : delta_aff.g[i] * delta_aff.lambda_g[i] + r_center.r_g[i]);
    residuals.t = uA.map((uAi, i) => uAi === null ? 0.0 : delta_aff.t[i] * delta_aff.lambda_t[i] + r_center.r_t[i]);
    residuals.y = lx.map((lxi, i) => lxi === null ? 0.0 : delta_aff.y[i] * delta_aff.lambda_y[i] + r_center.r_y[i]);
    residuals.z = ux.map((uxi, i) => uxi === null ? 0.0 : delta_aff.z[i] * delta_aff.lambda_z[i] + r_center.r_z[i]);
    const delta = computeSearchDirection(L, ipivOrD, variables, residuals);
    const { alphaP, alphaD } = getMaxStep(variables, delta);

    // Update the variables
    const fractionToBoundary = m > 0 ? 0.995 : 1.0;
    updateVariables(variables, delta, fractionToBoundary * alphaP, fractionToBoundary * alphaD);
  }

  // Return the solution and objective value
  const { f, r : residuals } = evalFunc(variables);
  const res = getResidualNorm(residuals);
  const gap = getMu(variables);
  return { x : variables.x, f, res, gap, iter };
}

function sanityCheck(condition, message) {
  if (!condition) throw new Error(message);
}

// Helper functions for linear algebra operations
const createVector = (n, initialValue = 0) => Array(n).fill(initialValue);
const createMatrix = (rows, cols, initialValue = 0) => Array(rows).fill().map(() => Array(cols).fill(initialValue));
const norm = (x) => Math.sqrt(dot(x, x));
const isVector = (x) => Array.isArray(x) && x.every(xi => typeof xi === 'number');
const assertIsVector = (x, name) => sanityCheck(isVector(x), 'Invalid input type: ' + name + ' must be an array. ' + name + ': ' + x);
const assertIsMatrix = (A) => sanityCheck(Array.isArray(A) && A.every(row => isVector(row)), 'Invalid input type: A must be a matrix. A: ' + A);
const assertAreEqualLength = (x, y) => sanityCheck(x.length === y.length, 'Invalid input shape: x and y must have the same length. x.length = ' + x.length + ', y.length = ' + y.length);

function setSubmatrix(M, X, startI, startJ) {
  const m = X.length;
  sanityCheck(M.length >= m + startI, 'Invalid submatrix row');
  for (let i = 0; i < m; i++) {
    const si = i + startI;
    const n = X[i].length;
    sanityCheck(M[si].length >= n + startJ, 'Invalid submatrix column');
    for (let j = 0; j < n; j++) {
      M[si][j + startJ] = X[i][j];
    }
  }
}

function setSubdiagonal(M, d, startRow, startCol) {
  sanityCheck(M.length >= d.length + startRow, 'Invalid submatrix row');
  d.forEach((value, i) => {
    M[i + startRow][i + startCol] = value;
  });
}

function diag(x) {
  assertIsVector(x, 'x');
  m = x.length;
  X = createMatrix(m, m);
  x.forEach((value, i) => {
    X[i][i] = value;
  });
  return X;
}

function assertAreEqualLengthVectors(x, y) {
  assertIsVector(x, 'x');
  assertIsVector(y, 'y');
  assertAreEqualLength(x, y);
}

function transpose(A) {
  assertIsMatrix(A);
  const m = A.length;
  const n = m > 0 ? A[0].length : 0;
  const B = createMatrix(n, m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      B[j][i] = A[i][j];
    }
  }
  return B;
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

function dot(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.reduce((sum, value, index) => sum + value * y[index], 0);
}
