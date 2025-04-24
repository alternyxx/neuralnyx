/* 
    custom wgsl file allowing javascript string interpolation :questionable:
    and since i have that why use pipeline constants :shrug:
    there are examples provided below the template used to provide an example
    of what the passed values could be, for exact values, check the generate_wgsl() in neuralnet.rs
    maybe i should switch to tera but that's like a whole engine :sob: :sob:
    it is also recommended to shift tab (remove indentation) the literals if youre debugging generated code
*/

const n_inputs = ${n_inputs};
const batch_size = ${batch_size};
const n_outputs = ${n_outputs};

struct Weights {
    ${i_weights}    // ie, weights0: array<array<f32, 12>, 9>
}

struct Biases {
    ${i_biases}    // ie, biases0: array<f32, 12>
}

struct Outputs {
    costs: array<f32, batch_size>,
    grad_weights: array<Weights, batch_size>,
    grad_biases: array<Biases, batch_size>,
}

@group(0) @binding(0) var<storage> X: array<array<f32, n_inputs>, batch_size>;
@group(0) @binding(1) var<storage> weights: Weights;
@group(0) @binding(2) var<storage> biases: Biases;
@group(0) @binding(3) var<storage> targets: array<array<f32, n_outputs>, batch_size>;
@group(0) @binding(4) var<storage, read_write> outputs: Outputs;
${storage}

@compute @workgroup_size(batch_size)
fn forward_pass(@builtin(global_invocation_id) id: vec3u) { 
    // i made this a template and grouped it because 
    // otherwise it causes a stack overflow
    ${forward}

    outputs.costs[id.x] = ${cost_function}(id.x);

    ${backpropagate}
}

// tanh is a built-in function

fn dtanh(al: f32) -> f32 {
    return 1.0 - (al * al);
}

fn sigmoid(zl: f32) -> f32 {
    return 1.0 / (1.0 + exp(-zl));
}

fn dsigmoid(al: f32) -> f32 {
    return al * (1.0 - al);
}

fn relu(zl: f32) -> f32 {
    return max(0.0, zl);
}

fn drelu(al: f32) -> f32 {
    return select(0.0, 1.0, al > 0.0);
}

fn leaky_relu(zl: f32) -> f32 {
    return max(0.01 * zl, zl);
}

fn dleaky_relu(al: f32) -> f32 {
    return 1.0;
}

// this is dumb but makes it a lot easier for code generation
fn linear(zl: f32) -> f32 {
    return zl;
}

fn dlinear(al: f32) -> f32 {
    return 1.0;
}

fn softmax_activation() {
    // find the highest
    var highest = al${n_al}[0];
    for (var i = 1; i < n_outputs; i++) {
        highest = max(highest, al${n_al}[i]);
    }
    
    // calculate e^zl_i
    var sum = 1.0e-7;
    for (var i = 0; i < n_outputs; i++) {
        let tmp = exp(al${n_al}[i] - highest);
        al${n_al}[i] = tmp;
        sum += tmp; 
    }

    // e^zl_i / sum(e^zl)
    for (var i = 0; i < n_outputs; i++) {
        al${n_al}[i] /= sum;
    }
}

// taking an array<f32, n_outputs> instead of 
// just an id can cause stack overflows.
fn mean_squared_error(id: u32) -> f32 {
    var cost = 0.0;
    
    for (var i = 0; i < n_outputs; i++) {
        cost += pow(al${n_al}[i] - targets[id][i], 2.0);
    }

    return cost / f32(n_outputs);
}

fn binary_cross_entropy(id: u32) -> f32 {
    var cost = 0.0;
    
    for (var i = 0; i < n_outputs; i++) {
        cost += targets[id][i] * log(max(al${n_al}[i], 1.0e-7))
                + (1.0 - targets[id][i]) * log(max(1.0 - al${n_al}[i], 1.0e-7));
    }

    return -cost / f32(n_outputs);
}

fn categorial_cross_entropy(id: u32) -> f32 {
    var cost = 0.0;
    
    for (var i = 0; i < n_outputs; i++) {
        cost += targets[id][i] * log(max(al${n_al}[i], 1.0e-7));
    }

    return -cost;
}