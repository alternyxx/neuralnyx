// will gradually abstract parts of generate_wgsl to here so it isnt a mess there
use textwrap::indent;
use indoc::formatdoc;

/* all functions below are functions that return wgsl code */
pub(crate) fn generate_cas(var: &str, v: &str, indentation: &str) -> String {
    indent(&formatdoc! { /*wgsl*/"
        let v = {v};
        var old_bits: u32 = atomicLoad(&{var});
        loop {{
            let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + v);

            let result = atomicCompareExchangeWeak(&{var}, old_bits, new_bits);
            if (result.exchanged) {{
                break;
            }}

            old_bits = result.old_value;
        }}
    "}, indentation)
}