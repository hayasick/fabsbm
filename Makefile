
RES_PATH = results

data = balanced #unbalanced
methods = ICL ICLO VB VAB FVAB FABVB
Ns = 100 200 400 800
K = 4
seed_max = 9

export OMP_NUM_THREADS := 1

all: $(RES_PATH) \
	$(foreach s,$(shell seq -w -s' ' 0 $(seed_max)),\
	$(foreach d,$(data),\
	$(foreach n,$(Ns),\
	$(foreach m,$(methods),\
		$(RES_PATH)/$(d)_$(n)_$(K)_$(s)_$(m)\
	)))) 

$(RES_PATH):
	mkdir $@

$(RES_PATH)/%:
	python run.py $(subst _, ,$*) > $@

summary:
	cat $(RES_PATH)/* > $@

plot: summary
	Rscript plot.R

clean:
	find $(RES_PATH)/ -size  0 -print0 |xargs -0 rm
