# H329 resource measurement correction, before campaign fits

The first controller sampled three one-second mpstat records and took the minimum
per core/sibling. It admitted one core and stopped before any campaign fit. That
used a stricter transient minimum than the owner's >=85% idle requirement.

Correct the implementation to one complete five-second mpstat interval, checking
each assigned core and its sibling separately against the same >=85% threshold.
Take one corrected launch probe; if the complete36 forecast still misses20:43,
close INCOMPLETE. No repeated probe selection, threshold relaxation or extra cores.

Preserve the original failed unit/log/forecast. Reuse the exact BASE regression,
input hashes and all six fixed synthetic fixture fits; no repeats. Original
preregistration, DART parameters, model streams,36-cell design, statistical rules,
20:45compute stop and21:00close remain unchanged. Previous unit must be stopped
before the corrected controller launches. This is a resource implementation fix,
not authorization to run a reduced experiment or reset the deadline.
