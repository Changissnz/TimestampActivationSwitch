# TimestampActivationSwitch

This is a data structure that can be thought of as a switch-like object
defined to function over a time period `(t,l)`. At each time `t0`, calculates
a value `pv` in the range `[0,1]` that is the probability value by `pFunc` (see
`class<TimeIntervalPD>` for more details). Then it will apply the function `bFunc`
onto `pv` to yield . Termination midway through `(t,l)` is enabled by the `tFunc`
variable.
