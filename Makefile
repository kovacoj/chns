SHELL := /bin/zsh

.PHONY: clean build run run-gui run-ch run-chns run-chns-two run-parallel help

clean:
	@setopt nullglob; \
	for dir in output; do \
		rm -f $$dir/*.{out,log,fls,blg,fdb_latexmk,aux,bbl,bcf,run.xml,synctex.gz}; \
	done

FIREDRAKE_IMAGE = firedrakeproject/firedrake:2025.4.2

build:
	docker pull $(FIREDRAKE_IMAGE)

run:
	bash ./run_firedrake_container bash

run-gui:
	bash ./run_firedrake_container bash

run-ch:
	bash ./run_firedrake_container python3 src/ch.py

run-chns:
	bash ./run_firedrake_container python3 src/chns.py --benchmark single_bubble

run-chns-two:
	bash ./run_firedrake_container python3 src/chns.py --benchmark two_bubbles

run-parallel:
	bash ./run_firedrake_container mpiexec -n 2 python3 src/parallel.py

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build         Pull the supported Firedrake image"
	@echo "  run           Open an interactive Firedrake shell"
	@echo "  run-gui       Alias for 'run' (the helper already wires X11 mounts)"
	@echo "  run-ch        Run the periodic Cahn-Hilliard benchmark"
	@echo "  run-chns      Run the canonical single-bubble CHNS benchmark"
	@echo "  run-chns-two  Run the canonical two-bubble CHNS benchmark"
	@echo "  run-parallel  Run the experimental MPI CHNS variant"
	@echo "  clean         Remove LaTeX auxiliary files from output/"
	@echo "  help          Show this help message"
