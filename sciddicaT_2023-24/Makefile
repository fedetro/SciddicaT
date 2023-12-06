# definisce la macro CPPC
ifndef CPPC
	CPPC=g++
endif

# STRESS_TEST_R = 1

ifdef STRESS_TEST_R
	HDR=./data/stress_test_R_header.txt
	DEM=./data/stress_test_R_dem.txt
	SRC=./data/stress_test_R_source.txt
	OUT=./strss_test_R_output_OpenMP
	OUT_SERIAL=./stress_test_R_output_serial
	STEPS=4000
else
	HDR=./data/tessina_header.txt
	DEM=./data/tessina_dem.txt
	SRC=./data/tessina_source.txt
	OUT=./tessina_output_OpenMP
	OUT_SERIAL=./tessina_output_serial
	STEPS=4000
endif

# definisce le macro contenenti i nomei degli eseguibili
# e il numero di thread omp per la versione parallela
NT = 2 # numero di threads OpenMP
EXEC = sciddicaT
EXEC_SERIAL = sciddicaTserial

# definisce il target di default, utile in
# caso di invocazione di make senza parametri
default:$(EXEC)

# compila le versioni seriale e OpenMP
$(EXEC): sciddicaT.cpp
	$(CPPC) $^ -o $@ -fopenmp -O3

$(EXEC_SERIAL): sciddicaT.cpp
	$(CPPC) $^ -o $@ -O3

# esegue la simulazione OpenMP
run:
	./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

runNThreads:
	OMP_NUM_THREADS=$(NT) ./$(EXEC) $(HDR) $(DEM) $(SRC) $(OUT) $(STEPS) &&  md5sum $(OUT) && cat $(HDR) $(OUT) > $(OUT).qgis && rm $(OUT)

# esegue la simulazione seriale 
run_serial:
	./$(EXEC_SERIAL) $(HDR) $(DEM) $(SRC) $(OUT_SERIAL) $(STEPS) &&  md5sum $(OUT_SERIAL) && cat $(HDR) $(OUT_SERIAL) > $(OUT_SERIAL).qgis && rm $(OUT_SERIAL)

# elimina l'eseguibile, file oggetto e file di output
clean:
	rm -f $(EXEC) $(EXEC_SERIAL) *.o *output*

# elimina file oggetto e di output
wipe:
	rm -f *.o *output*
