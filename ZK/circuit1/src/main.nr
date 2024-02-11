fn main(x: [Field;8], y: [Field;8]) -> pub [Field;8] {
    let a = x[0] / 2 + y[0] / 2;
    let b = x[1] / 2 + y[1] / 2;
    let c = x[2] / 2 + y[2] / 2;
    let d = x[3] / 2 + y[3] / 2;
    let e = x[4] / 2 + y[4] / 2;
    let f = x[5] / 2 + y[5] / 2;
    let g = x[6] / 2 + y[6] / 2;
    let h = x[7] / 2 + y[7] / 2;
    [a,b,c,d,e,f,g,h]
}