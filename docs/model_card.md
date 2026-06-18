# Model Card - SIMAP-IA

## Propósito

Clasificar el riesgo de falla de una máquina industrial bajo condiciones operativas específicas.

## Dataset

AI4I 2020 Predictive Maintenance Dataset.

## Alcance

El sistema estima probabilidad de falla. No predice fecha exacta de falla ni vida útil restante.

## Variables de entrada

- Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

## Variables excluidas

- UDI
- Product ID
- TWF
- HDF
- PWF
- OSF
- RNF

Estas variables no deben entrar como predictores del modelo principal.
