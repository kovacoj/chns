
# Self-propelled droplet

This repo contains Firedrake experiments for Cahn-Hilliard and Cahn-Hilliard-Navier-Stokes simulations. The current canonical CHNS benchmark lives in `src/chns.py` and uses a Taylor-Hood velocity-pressure pair on a rectangle for rising-bubble studies. Other solver files explore pressure-robust, barycentrically refined, parallel, and many-bubble variants.

<p align="center">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bublina.gif" alt="bublina">
  <img width="45%" src="https://github.com/jk-dot/chns/blob/main/report/graphics/bubliny.gif" alt="bubliny">
</p>


# TODO
- [ ] error estimation and mesh adaptivity
- [ ] pressure-robust discretization study
