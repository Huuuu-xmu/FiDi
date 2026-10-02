# Contributing

Describe the scientific reason for model or metric changes and update the
corresponding documentation. Keep historical evidence and newly generated
results identified separately. Record the configuration, seed, source identity
and environment for each run.

Run the unit tests and the synthetic smoke command after changes to the model
or workflow. Keep clinical images, participant information, private manifests,
checkpoints and credentials outside Git. Report failures with synthetic examples
and sanitized logs where possible.

Changes to metric definitions require a version increment and new result tables;
they must not silently change the meaning of previously reported scores.
