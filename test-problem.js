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

    if (false) {
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

document.getElementById("test-button").addEventListener("click", solveTestProblem);
