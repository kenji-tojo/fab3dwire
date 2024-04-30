# Automatic Jig Generation

Put wavefront object file named `input.obj` under a folder `asset/<name>/`

```bash
cargo run --example 0_split_input_loop_into_two --release <name>
```

This will generate `a.obj` and `b.obj` in the `asset/<name>` folder.


Next, generate ribbons with 

```bash
cargo run --example 1_generate_ribbon --release <name>/a
```

Finally, generate pillars with 


```bash
cargo run --example 2_generate_pillars --release <name>/a
```