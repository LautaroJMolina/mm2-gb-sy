# mm2-gb-sy
mm2-gb-sy es una migración de [mm2-gb](https://github.com/Minimap2onGPU/mm2-gb "mm2-gb") a SYCL. Ofreciendo mejor portabilidad con mínimo impacto de rendimiento.

## Requerimientos
OS: Linux
Se requiere un compilador de SYCL que se adhiere al estándar SYCL2020. Recomendamos particularmente [AdaptiveCPP](https://github.com/AdaptiveCpp/AdaptiveCpp "AdaptiveCPP").

## Instalación
Clonar el repositorio y compilar con el comando make especificando el nivel de depuración deseado:
```
git clone --recursive git@github.com:LautaroJMolina/mm2-gb-sy.git mm2-gb-sy
cd mm2-gb-sy

# DEBUG nivel:
#   <empty>   : solo se imprime la salida de minimap2
#   info      : imprime la salida de estadísticas de kernel de encadenado. (Excepto throughput en pares de anchors/s)
#   analyze   : calcula throughtput de kernel en pares de anchors / s. (requiere sincronización adicional)
#   verbose   : imprime información de lanzamiento / depuración de kernels.

# Ejemplo de compilación
make DEBUG=analyze
```
A diferencia de mm2-gb, no es necesario especificar el fabricante ni arquitectura del dispositivo durante la compilación.

## Uso
El uso de mm2-gb-sy es idéntico al de mm2-gb, por lo cual no difiere mucho del uso de minimap2. La flag `--gpu-chain` activa el encadenado en GPU. La flag `--gpu-cfg <filename.json>` especifica el archivo json de configuración de GPU.
Por ahora mm2-gb y por lo tanto mm2-gb-sy solo soportan un único hilo para el procesamiento en CPU (-t 1).

Un ejemplo de mapeo con GPU: 
```
./minimap2 -t 1 --gpu-chain --gpu-cfg gpu/gpu_config.json test/MT-human.fa test/MT-orang.fa > mm2-gb_out.paf
```

### Configuración GPU
mm2-gb provee archivos de configuración ejemplo para algunos dispositivos, adicionalmente incluimos los archivos utilizados para experimentos de rendimiento (`gpu/3070_config.json`, `gpu/gfx1034_config.json`).

Para más información sobre los archivos de configuración referirse a la documentación de [mm2-gb](https://github.com/Minimap2onGPU/mm2-gb "mm2-gb").

## Verificación de resultados
Para verificar los resultados, utilizar paftools (`misc/paftools.js`) con el subcomando pafcmp. Requiere [k8](https://github.com/attractivechaos/k8 "k8").