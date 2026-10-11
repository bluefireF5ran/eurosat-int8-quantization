# Hipótesis (escritas el 11-oct-2026, antes de medir nada)

Pregunta: ¿cuánta precisión pierde un clasificador de imágenes de satélite al
cuantizarlo a 8 bits, y cuánto gana en tamaño y velocidad?

1. **Precisión:** perderá algo, pero poco. Si el original acierta en torno al
   95 %, el cuantizado quedará en torno al 85-90 %.
2. **Tamaño:** unas 4 veces más pequeño, porque cada peso pasa de 32 a 8 bits.
3. **Velocidad:** irá mucho más rápido.
