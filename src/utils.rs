use std::collections::HashMap;

pub(crate) fn flatten2d(vec2d: &Vec<Vec<f32>>) -> Vec<f32> {
    vec2d.iter().flatten().copied().collect::<Vec<f32>>()
}

pub(crate) fn flatten3d(vec3d: &Vec<Vec<Vec<f32>>>) -> Vec<f32> {
    vec3d.iter().flatten().flatten().copied().collect::<Vec<f32>>()
}

// this function is created because i want js/ts template literals and
// pipeline constants aren't enough
// also im not gonna write a whole parser just so this can ignore comments xd
// it is also worth nothing that it is VERY lazily made
pub(crate) fn template_wgsl(wgsl: &str, literals: &HashMap<String, String>) -> String {
    let mut templating = false;
    let mut template_variable: String = String::new();
    let mut templated_wgsl: String = String::new();

    for char in wgsl.chars() {
        // in the process of templating
        if templating {
            if char == '}' {
                templated_wgsl += literals.get(&template_variable.to_string())
                    .unwrap_or_else(|| panic!("\n{} wasn't given\n", template_variable.to_string()));

                template_variable = String::new();
                templating = false;
            } else if char == '{' {
                continue
            } else {
                template_variable += &char.to_string();    
            }

            continue
        } else if char == '$' {
            templating = true;
        } else {
            templated_wgsl += &char.to_string();
        }
    }
    
    // println!("{templated_wgsl}"); // lazy debugging :P
    templated_wgsl
}