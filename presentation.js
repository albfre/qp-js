// Functions realting to parsing of optimization problem
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
  const variableSet = new Set();
  validatePowers(terms, [1, 2]);
  for (const term of terms) {
    addOrInsert(term.power === 1 ? linearCoefficients : quadraticCoefficients, term);
    variableSet.add(term.variable);
  }

  for (const key in quadraticCoefficients) {
    if (quadraticCoefficients[key] <= 0) {
      throw new Error('Variable "' + key + '" has negative quadratic coefficient.');
    }
  }
  const variables = Array.from(variableSet).sort();
  const m = variables.length;
  const Q = createMatrix(m, m);
  for (const key in quadraticCoefficients) {
    const index = variables.indexOf(key);
    Q[index][index] = 2 * quadraticCoefficients[key];
  }

  const c = createVector(m);
  for (const key in linearCoefficients) {
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
  let Acs = [];
  let xcs = [];
  for (const constraint of constraints) {
    const { coefficients, separator, rhs } = parseConstraint(constraint);
    if (coefficients.length === 1) {
      xcs.push({ coefficients, separator, rhs });
    }
    else {
      Acs.push({ coefficients, separator, rhs });
    }
  }

  function createConstraints(cs) {
    const A = createMatrix(cs.length, m);
    const l = createVector(cs.length);
    const u = createVector(cs.length).fill(null);
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      const sign = c.separator === '<=' ? -1 : 1;
      for (const v in c.coefficients) {
        if (!variables.includes(v)) {
          throw new Error('Constraint variable "' + v + '" is not included in the objective.');
        }
        A[i][variables.indexOf(v)] = sign * c.coefficients[v];
      }
      l[i] = sign * c.rhs;
      if (c.separator === '=') {
        u[i] = sign * c.rhs;
      }
    }
    return { A, l, u };
  }

  function createBounds(cs) {
    const l = createVector(m).fill(null);
    const u = createVector(m).fill(null);
    for (let i = 0; i < cs.length; i++) {
      const c = cs[i];
      if (c.coefficients.length !== 1) {
        throw new Error('Not a bound: ' + c.coefficients);
      }
      const sign = c.separator === '<=' ? -1 : 1;
      for (const v in c.coefficients) {
        if (!variables.includes(v)) {
          throw new Error('Constraint variable "' + v + '" is not included in the objective.');
        }
        const coefficient = c.coefficients[v];
        const index = variables.indexOf(v);
        if (coefficient !== 0.0) {
          const isPositive = coefficient > 0.0;
          if (c.separator !== (isPositive ? '>=' : '<=')) {
            l[index] = c.rhs / coefficient;
          }
          if (c.separator !== (isPositive ? '<=' : '>=')) {
            u[index] = c.rhs / coefficient;
          }
        }
      }
    }
    return { l, u };
  }

  const { A, l : lA, u : uA } = createConstraints(Acs);
  const { l : lx, u : ux } = createBounds(xcs);

  return { A, lA, uA, lx, ux };
}

// Functions relating html page
function solveQP(Q, c, A, lA, uA, lx, ux, variables = []) {
  let solutionElement = document.getElementById("solution");
  
  try {
    const start = performance.now();
    const {x, f, res, gap, iter} = interiorPointQP(Q, c, A, lA, uA, lx, ux);
    const end = performance.now();

    let tableStr = '<table>';
    const addRow = (str, val) => tableStr += `<tr><td class="no-wrap">${str}</td><td>${val}</td></tr>\n`;

    addRow('Objective value', f);
    addRow('Number of iterations', iter);
    addRow('Residual', res);
    addRow('Gap', gap);
    addRow('Elapsed time', `${end - start} milliseconds`);
    for (let i = 0; i < x.length; i++) {
      addRow(variables.length === x.length ? variables[i] : `x${i}`, x[i]);
    }
    addRow('Variable vector', x.join(', '));
    tableStr += '</table>';

    solutionElement.innerHTML = tableStr;

  } catch (error) {
    solutionElement.innerHTML = `Error ${error.lineNumber}: ${error.message}`;
  }
}

function solveQP_old(Q, c, Aeq, beq, Aineq, bineq, variables = []) {
  const n = Q.length;
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
  let lx = createVector(n, null);
  let ux = createVector(n, null);
  return solveQP(Q, c, A, lA, uA, lx, ux, variables);
}

function solve() {
  const objective = document.getElementById("objective").value;
  const table = document.getElementById("optimization-problem");
  let constraints = [];
  for (let i = 0; i < table.rows.length; i++) {
    const constraint = document.getElementById(`constraint-${i}`);
    if (constraint) {
      constraints.push(constraint.textContent);
    }
  }
  try {
    const { Q, c, variables } = parseObjective(objective);
    const { A, lA, uA, lx, ux } = parseConstraints(variables, constraints)
    solveQP(Q, c, A, lA, uA, lx, ux, variables);
  } catch (error) {
    let solutionElement = document.getElementById("solution");
    solutionElement.innerHTML = `Error ${error.lineNumber}: ${error.message}`;
  }
}

function clear() {
  const table = document.getElementById("optimization-problem");
  for (let i = table.rows.length; i > 2; i--) {
    table.deleteRow(i - 1);
  }
  solutionElement = document.getElementById("solution");
  solutionElement.innerHTML = '';
}

function addConstraint() {
  const constraint = document.getElementById("constraint");
  if (constraint.value.trim().length > 0) {
    const table = document.getElementById("optimization-problem");
    const m = table.rows.length;
    const row = table.insertRow(m);
    row.innerHTML = `<td></td><td id="constraint-${m}">` + constraint.value + '</td>';
  }
  constraint.value = '';
}

document.getElementById("solve-button").addEventListener("click", solve);
document.getElementById("clear-button").addEventListener("click", clear);
document.getElementById("add-constraint-button").addEventListener("click", addConstraint);
