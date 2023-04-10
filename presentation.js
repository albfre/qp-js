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

// Functions relating html page
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
