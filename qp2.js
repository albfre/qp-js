function interiorPointQP(H, c, Aeq, beq, Aineq, bineq, tol=1e-8, maxIter=100) {
  /* minimize 0.5 x' H x + c' x
   *   st    Aeq x = beq
   *         Aineq x >= bineq
   */
   const n = H.length;
   const mEq = Aeq.length;
   const mIneq = Aineq.length;
   const A = Aeq.concat(Aineq);
   let lA = zeroVector(mEq + mIneq);
   let uA = zeroVector(mEq + mIneq);
   for (let i = 0; i < mEq; ++i) {
     lA[i] = beq[i];
     uA[i] = beq[i];
   }
   for (let i = 0; i < mIneq; ++i) {
     lA[i + mEq] = bineq[i];
     uA[i + mEq] = null;
   }
   let lx = zeroVector(n).fill(null);
   let ux = zeroVector(n).fill(null);
   return interiorPointQP2(H, c, A, lA, uA, lx, ux, tol, maxIter);
}

function interiorPointQP2(H, c, A, lA, uA, lx, ux, tol=1e-8, maxIter=100) {
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

  function evalSlackResiduals(v, mu) {
    const mu_m = new Array(m).fill(mu);
    const mu_n = new Array(n).fill(mu);
    const r_g = zeroIfNull(subtract(multiply(v.g, v.lambda_g), mu_m), lA); // G Lambda_g e - mu e
    const r_t = zeroIfNull(subtract(multiply(v.t, v.lambda_t), mu_m), uA); // T Lambda_t e - mu e
    const r_y = zeroIfNull(subtract(multiply(v.y, v.lambda_y), mu_n), lx); // Y Lambda_y e - mu e
    const r_z = zeroIfNull(subtract(multiply(v.z, v.lambda_z), mu_n), ux); // Z Lambda_z e - mu e
    return { r_g, r_t, r_y, r_z };
  }

  // Define the function for evaluating the objective and constraints and compute the residuals
  function evalFunc(v, mu) {
    const Hx = matrixTimesVector(H, v.x);
    const Ax = matrixTimesVector(A, v.x);

    // Objective
    const f = 0.5 * dot(v.x, Hx) + dot(c, v.x); // 0.5 x' H x + c' x

    // Residuals
    const r = structuredClone(v);

    r.x = add(Hx, add(c, subtract(v.lambda_z, v.lambda_y))); // Hx + c + A' lambda_A - lambda_y + lambda_z
    if (m > 0) {
      r.x = add(r.x, matrixTimesVector(AT, v.lambda_A));
    }

    r.s = subtract(v.lambda_t, add(v.lambda_A, v.lambda_g)); // -lambda_A - lambda_g + lambda_t
    ({ r_g : r.g, r_t : r.t, r_y : r.y, r_z : r.z } = evalSlackResiduals(v, mu));
    r.lambda_A = subtract(Ax, v.s); // Ax - s
    r.lambda_g = subtractOrZeroIfNull(subtract(v.s, v.g), lA); // s - g - lA
    r.lambda_t = subtractOrZeroIfNull(add(v.s, v.t), uA); // s + t - uA
    r.lambda_y = subtractOrZeroIfNull(subtract(v.x, v.y), lx); // x - y - lx
    r.lambda_z = subtractOrZeroIfNull(add(v.x, v.z), ux); // x + z - ux

    return { f, r };
  }

  // Construct the KKT system
  /*  [ H + Y^-1 Lambda_y + Z^-1 Lambda_z  A'                                  ]
   *  [ A                                  -(G^-1 Lambda_g + T^-1 Lambda_t)^-1 ]
  */
  const KKT = zeroMatrix(n + m, n + m);
  setSubmatrix(KKT, H, 0, 0);
  setSubmatrix(KKT, AT, 0, n);
  setSubmatrix(KKT, A, n, 0);
  const hDiag = H.map((row, i) => row[i]);

  function updateMatrix(v) {
    const newHDiag = hDiag.map((val, i) => val + v.lambda_y[i] / v.y[i] + v.lambda_z[i] / v.z[i]);
    setSubdiagonal(KKT, newHDiag, 0, 0);
    const d = Array.from({ length: m }, (_, i) => -1.0 / (v.lambda_g[i] / v.g[i] + v.lambda_t[i] / v.t[i]));
    setSubdiagonal(KKT, d, n, n);
  }

  const zeroIfNull = (lambda, bound) => lambda = lambda.map((v, i) => bound[i] === null ? 0.0 : v);

  // Define the function for computing the search direction
  function computeSearchDirection(L, ipiv, v, r) {
    const yPart = divide(add(multiply(v.lambda_y, r.lambda_y), r.y), v.y); // Y^-1 (L_y r_lambda_y + r_y)
    const zPart = divide(subtract(multiply(v.lambda_z, r.lambda_z), r.z), v.z); // Z^-1 (L_z r_lambda_z - r_z)
    const rhs1 = add(add(r.x, yPart), zPart);
    
    const gtPart = add(divide(v.lambda_g, v.g), divide(v.lambda_t, v.t)); // G^-1 L_g  + T^-1 L_t
    const gPart = divide(add(multiply(v.lambda_g, r.lambda_g), r.g), v.g); // G^-1 (L_g r_lambda_g + r_g)
    const tPart = divide(subtract(multiply(v.lambda_t, r.lambda_t), r.t), v.t); // T^-1 (L_t r_lambda_t - r_t)
    const sPart = add(add(tPart, gPart), r.s);
    const rhs2 = add(r.lambda_A, divide(sPart, gtPart));

    const rhs = negate(rhs1.concat(rhs2));

    // Solve the KKT system
    const delta = solveUsingFactorization(L, ipiv, rhs);

    // Extract the search direction components
    const d = structuredClone(v);

    d.x = delta.slice(0, n);
    d.lambda_A = delta.slice(n, n + m);
    d.s = divide(subtract(d.lambda_A, sPart), gtPart);
    d.lambda_g = zeroIfNull(negate(divide(add(multiply(v.lambda_g, add(d.s, r.lambda_g)), r.g), v.g)), lA);
    d.lambda_t = zeroIfNull(divide(subtract(multiply(v.lambda_t, add(d.s, r.lambda_t)), r.t), v.t), uA);
    d.lambda_y = zeroIfNull(negate(divide(add(multiply(v.lambda_y, add(d.x, r.lambda_y)), r.y), v.y)), lx);
    d.lambda_z = zeroIfNull(divide(subtract(multiply(v.lambda_z, add(d.x, r.lambda_z)), r.z), v.z), ux);
    d.g = zeroIfNull(negate(divide(add(r.g, multiply(g, d.lambda_g)), v.lambda_g)), lA);
    d.t = zeroIfNull(negate(divide(add(r.t, multiply(t, d.lambda_t)), v.lambda_t)), uA);
    d.y = zeroIfNull(negate(divide(add(r.y, multiply(y, d.lambda_y)), v.lambda_y)), lx);
    d.z = zeroIfNull(negate(divide(add(r.z, multiply(z, d.lambda_z)), v.lambda_z)), ux);

    return d;
  }

  // Initialize primal and dual variables
  const x = filledVector(n, 1.0);
  const y = filledVector(n, 1.0); // Slack for x
  const z = filledVector(n, 1.0); // Slack for x
  const lambda_y = zeroIfNull(filledVector(n, 1.0), lx);
  const lambda_z = zeroIfNull(filledVector(n, 1.0), ux);

  const s = filledVector(m, 1.0); // Slack variables for inequality constraints
  const g = filledVector(m, 1.0); // Slack for s
  const t = filledVector(m, 1.0); // Slack for s
  const lambda_A = filledVector(m, 1.0);
  const lambda_g = zeroIfNull(filledVector(m, 1.0), lA);
  const lambda_t = zeroIfNull(filledVector(m, 1.0), uA);
  const variablesAndMultipliers = { x, s, g, t, y, z, lambda_A, lambda_g, lambda_t, lambda_y, lambda_z };
  
  function getMu(v) {
    return m > 0 ? (dot(v.g, v.lambda_g) + dot(v.t, v.lambda_t)) / m + (dot(v.y, v.lambda_y) + dot(v.z, v.lambda_z)) / n : 0.0;
  }

  function getResidualNorm(r) {
    return norm([].concat(r.x, r.s, r.g, r.t, r.y, r.z, r.lambda_A, r.lambda_g, r.lambda_t, r.lambda_y, r.lambda_z));
  }

  // Define the function for computing the step size
  function getMaxStep(vv, dvv) {
    const getMaxStepSingle = (v, dv) => v.reduce((m, value, index) => dv[index] < 0 ? Math.min(-value / dv[index], m) : m, 1.0);
    const alphaG = getMaxStepSingle(vv.g, dvv.g);
    const alphaT = getMaxStepSingle(vv.t, dvv.t);
    const alphaY = getMaxStepSingle(vv.y, dvv.y);
    const alphaZ = getMaxStepSingle(vv.z, dvv.z);
    const alphaP = Math.min(alphaG, alphaT, alphaY, alphaZ);

    const alphaLambdaG = getMaxStepSingle(vv.lambda_g, dvv.lambda_g);
    const alphaLambdaT = getMaxStepSingle(vv.lambda_t, dvv.lambda_t);
    const alphaLambdaY = getMaxStepSingle(vv.lambda_y, dvv.lambda_y);
    const alphaLambdaZ = getMaxStepSingle(vv.lambda_z, dvv.lambda_z);
    const alphaD = Math.min(alphaLambdaG, alphaLambdaT, alphaLambdaY, alphaLambdaZ);
    
    return { alphaP, alphaD };
  }

  // Perform the interior point optimization
  let iter = 0;
  for (; iter < maxIter; iter++) {
    const { f, r : residuals } = evalFunc(variablesAndMultipliers, 0.0);

    // Check the convergence criterion
    const residualNorm = getResidualNorm(residuals);
    const mu = getMu(variablesAndMultipliers);
    console.log(`${iter}. f: ${f}, res: ${residualNorm}, gap: ${mu}`)
    if (residualNorm <= tol && mu <= tol) {
      break;
    }

    // Update and factorize KKT matrix
    updateMatrix(variablesAndMultipliers);
    const [L, ipiv] = symmetricIndefiniteFactorization(KKT);

    // Use the predictor-corrector method

    // Compute affine scaling step
    const delta_aff = computeSearchDirection(L, ipiv, variablesAndMultipliers, residuals);
    const { alphaP : alphaP_aff, alphaD : alphaD_aff } = getMaxStep(variablesAndMultipliers, delta_aff);

    const v_aff = structuredClone(variablesAndMultipliers);
    vectorPlusEqScalarTimesVector(v_aff.g, alphaP_aff, delta_aff.g);
    vectorPlusEqScalarTimesVector(v_aff.t, alphaP_aff, delta_aff.t);
    vectorPlusEqScalarTimesVector(v_aff.y, alphaP_aff, delta_aff.y);
    vectorPlusEqScalarTimesVector(v_aff.z, alphaP_aff, delta_aff.z);
    vectorPlusEqScalarTimesVector(v_aff.lambda_g, alphaD_aff, delta_aff.lambda_g);
    vectorPlusEqScalarTimesVector(v_aff.lambda_t, alphaD_aff, delta_aff.lambda_t);
    vectorPlusEqScalarTimesVector(v_aff.lambda_y, alphaD_aff, delta_aff.lambda_y);
    vectorPlusEqScalarTimesVector(v_aff.lambda_z, alphaD_aff, delta_aff.lambda_z);
    const muAff = getMu(v_aff);

    // Compute aggregated centering-corrector direction
    const sigma = mu > 0 ? Math.pow(muAff / mu, 3.0) : 0;

    const r_center = evalSlackResiduals(variablesAndMultipliers, sigma * mu);
    residuals.g = zeroIfNull(add(multiply(delta_aff.g, delta_aff.lambda_g), r_center.r_g), lA);
    residuals.t = zeroIfNull(add(multiply(delta_aff.t, delta_aff.lambda_t), r_center.r_t), uA);
    residuals.y = zeroIfNull(add(multiply(delta_aff.y, delta_aff.lambda_y), r_center.r_y), lx);
    residuals.z = zeroIfNull(add(multiply(delta_aff.z, delta_aff.lambda_z), r_center.r_z), ux);
    const delta = computeSearchDirection(L, ipiv, variablesAndMultipliers, residuals);
    const { alphaP, alphaD } = getMaxStep(variablesAndMultipliers, delta);

    // Update the variables
    const fractionToBoundary = m > 0 ? 0.995 : 1.0;
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.x, fractionToBoundary * alphaP, delta.x);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.s, fractionToBoundary * alphaP, delta.s);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.g, fractionToBoundary * alphaP, delta.g);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.t, fractionToBoundary * alphaP, delta.t);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.y, fractionToBoundary * alphaP, delta.y);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.z, fractionToBoundary * alphaP, delta.z);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.lambda_A, fractionToBoundary * alphaD, delta.lambda_A);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.lambda_g, fractionToBoundary * alphaD, delta.lambda_g);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.lambda_t, fractionToBoundary * alphaD, delta.lambda_t);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.lambda_y, fractionToBoundary * alphaD, delta.lambda_y);
    vectorPlusEqScalarTimesVector(variablesAndMultipliers.lambda_z, fractionToBoundary * alphaD, delta.lambda_z);
  }

  // Return the solution and objective value
  const { f, r : residuals } = evalFunc(variablesAndMultipliers, 0.0);
  const res = getResidualNorm(residuals);
  const gap = getMu(variablesAndMultipliers);
  return { x, f, res, gap, iter };
}

function sanityCheck(condition, message) {
  if (!condition) throw new Error(message);
}

// Helper functions for linear algebra operations
const filledVector = (n, v) => new Array(n).fill(v);
const zeroVector = (n) => filledVector(n, 0.0);
const filledMatrix = (m, n, v) => new Array(m).fill().map(() => new Array(n).fill(v));
const zeroMatrix = (m, n) => filledMatrix(m, n, 0.0);
const norm = (x) => Math.sqrt(dot(x, x));
const isVector = (x) => Array.isArray(x) && x.every(xi => typeof xi === 'number');
const assertIsVector = (x, name) => sanityCheck(isVector(x), 'Invalid input type: ' + name + ' must be an array. ' + name + ': ' + x);
const assertIsMatrix = (A) => sanityCheck(Array.isArray(A) && A.every(row => isVector(row)), 'Invalid input type: A must be a matrix. A: ' + A);
const assertAreEqualLength = (x, y) => sanityCheck(x.length === y.length, 'Invalid input shape: x and y must have the same length. x.length = ' + x.length + ', y.length = ' + y.length);

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
  assertAreEqualLength(x, y);
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

function multiply(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value * y[index]);
}

function divide(x, y) {
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

function subtract(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.map((value, index) => value - y[index]);
}

function subtractOrZeroIfNull(x, y) {
  assertAreEqualLength(x, y);
  return x.map((value, index) => y[index] === null ? 0.0 : value - y[index]);
}

function dot(x, y) {
  assertAreEqualLengthVectors(x, y);
  return x.reduce((sum, value, index) => sum + value * y[index], 0);
}
