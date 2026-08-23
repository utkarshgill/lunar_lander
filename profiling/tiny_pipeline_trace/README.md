# tinygrad PPO pipeline trace

This trace comes from the controlled 10,000-sample PPO update on CPU with `BEAM=1`.

The lazy loss and optimizer graph contains 1250 UOps and 68 requested outputs. Realization produces 150 program calls from 65 unique rendered programs.

`pipeline.json` lists calls in TinyJit replay order. Each call records its buffer shapes, output slots, input slots, launch size, source file, and source hash. Each source entry records the applied schedule options.

The first two calls are the actor and critic input layers. Their generated program fuses the matrix product and bias addition. Later calls implement hidden layers, PPO loss, backward operations, gradient clipping, and Adam state updates.
