# Keith Briggs 2025-09-17 CRRM-2.0

# this gets the project folder name...
mkfile_path:=$(abspath $(lastword $(MAKEFILE_LIST)))
PRJ:=$(notdir $(patsubst %/,%,$(dir $(mkfile_path))))

CRRM_test:
	@mkdir -p img
	@bash use_most_recent_python.sh CRRM_example_13_timing_tests.py 0

all_examples:
	@bash run_all_examples.sh

a: # animation test
	mkdir -p ani
	mkdir -p mp4
	rm -f ani/*png
	@bash use_most_recent_python.sh CRRM_example_13_timing_tests.py a
	ffmpeg -y -loglevel quiet -framerate 15 -pattern_type glob -i 'ani/*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p mp4/crrm_animation.mp4
	xdg-open mp4/crrm_animation.mp4

# pip install snakeviz ...
profile:
	@mkdir -p img
	@bash use_most_recent_python.sh -m cProfile -o CRRM_profile.dat CRRM_example_13_timimg_tests.py 0 && snakeviz CRRM_profile.dat

timing_tests:
	@bash use_most_recent_python.sh CRRM_large_system_timing_tests.py

doc: doc/sphinx_source/conf.py doc/sphinx_source/index.rst img/RMa_pathloss_model.png img/UMa_pathloss_model.png img/UMi_pathloss_model.png img/InH_pathloss_model.png img/power_law_pathloss_model.png
	@bash use_most_recent_python.sh -m sphinx -b html doc/sphinx_source doc/sphinx_build # needs python 3.11 or higher

img/RMa_pathloss_model.png:
	python3 CRRM/RMa_pathloss_model_08.py
img/UMa_pathloss_model.png:
	python3 CRRM/UMa_pathloss_model_06.py
img/UMi_pathloss_model.png:
	python3 CRRM/UMi_pathloss_model_00.py
img/InH_pathloss_model.png:
	python3 CRRM/InH_pathloss_model_01.py
img/power_law_pathloss_model.png:
	python3 CRRM/power_law_pathloss_model_02.py

# backup in ${PRJ}.zip
zip: # distribution
	(cd tex; texclean)
	@echo "making a distribution zipfile of ${PRJ}/*"
	rm -f ~/tarfiles/"${PRJ}.zip"
	(cd ..; zip -r "tarfiles/${PRJ}.zip" "${PRJ}" -x@"${PRJ}"/cfg/zip_excludes_dist.txt)
	cp -p ~/tarfiles/"${PRJ}".zip ~/Dropbox/tarfiles/
	cp -p ~/tarfiles/"${PRJ}".zip ~/Dropbox/Keith-Ibrahim/
	cp -p ~/tarfiles/"${PRJ}".zip ~/Dropbox/Keith-Ahmed/

zip_full: # full
	(cd tex; texclean)
	@echo "making a ifull zipfile of ${PRJ}/*"
	rm -f ~/tarfiles/"${PRJ}_full.zip"
	(cd ..; zip -r "tarfiles/${PRJ}_full.zip" "${PRJ}" -x@"${PRJ}"/cfg/zip_excludes_full.txt)
	cp -p ~/tarfiles/"${PRJ}_full".zip ~/Dropbox/tarfiles/
