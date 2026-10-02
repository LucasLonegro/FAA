# Novel Alt Data Sources — Quant Workshop

Oct 1, 2026 · @Lucas

## How to read this

Each proposed source passed three tests: the raw data is free and public, it answers a question tied to a listed company or traded commodity, and a search found no vendor, fund or published study already running it for that question.

- **Public**: downloadable or scrapable without a paid licence or NDA.
- **Tradable question**: a named KPI (volumes, utilisation, production) that the market prices before official numbers land.
- **No known pipeline**: no commercial product, and no paper or blog using the same data for the same question. Adjacent work is noted honestly under each idea.

Novelty checks are web searches as of the date above, not a guarantee. Ask participants to re-run the check as their first exercise.

## Already taken

Most famous alt-data ideas are now commercial products, so participants should treat these areas as off-limits. Rows marked \* were confirmed in this research; the rest are general industry knowledge, not re-verified.

| Area | Already answered by |
| --- | --- |
| Oil storage, tanker loadings, cargo flows | Floating-roof shadow analytics; AIS/draught vendors (Kpler, Vortexa) |
| Smelters, steel, factory emissions | Smelter-activity products (SAVANT); Sentinel-5P NO2 and night-lights studies |
| Store and venue visits, parking lots | Phone-geolocation panels (Placer.ai, Advan); lot-count imagery |
| Ski resort visits\* | Placer.ai geofencing of individual resorts ([TownLift](https://townlift.com/2022/05/customer-cell-phone-data-indicates-record-visitation-at-park-city-mountain-for-the-21-22-ski-season/)) |
| EV charger utilisation\* | [Chargalytics](https://www.chargalytics.com/) (Europe, 37 countries), [Paren](https://www.paren.app/) (US fast charging) |
| Data center build-out\* | Air-permit generator trackers ([SAVRN](https://savrn.com/blog/data-center-permit-tracker)); construction imagery |
| Consumer spend, web, apps, hiring | Card panels, web-traffic and app-download panels, job-posting scrapes |
| Developer adoption\* | npm/PyPI download trackers pitched to investors ([Apify example](https://apify.com/scrapemint/package-adoption-tracker)) |
| Crops and harvest progress\* | Sentinel-1/2 harvest monitoring, including Brazilian cane ([EOSDA](https://eos.com/blog/sugarcane-harvest-monitoring-in-brazil-dates-and-areas/)) |

## Shortlist at a glance

Seven sources cleared the screen; numbers 1, 2 and 7 are the easiest to get working in a one-day workshop.

| # | Public source | Question it answers | Instruments | Build difficulty | Novelty confidence |
| --- | --- | --- | --- | --- | --- |
| 1 | Hospital-posted ER wait times | ER demand by hospital and market, intra-quarter | HCA, THC, UHS, CYH | Low: scrape every 30 min | High |
| 2 | USGS seismic catalog: quarry and mine blasts | Blasting cadence at named quarries and mines | VMC, MLM, coal miners | Low: one public API | High |
| 3 | Brazil ONS hourly generation by plant | Cane crush pace from bagasse power exports | ICE sugar No. 11, Brazilian mill equities | Medium | Medium-high |
| 4 | FAA Form 7460-1 obstruction filings | Pipeline of cranes, towers and tall buildings \~45 days ahead | Tower REITs, construction and equipment names | Medium | Medium (wind turbines taken) |
| 5 | ADS-B tracks of offshore helicopters | Crew-change and maintenance activity per platform | Offshore producers, drillers, helicopter operators | Medium | Medium |
| 6 | Sentinel-2 SWIR over heap-leach pads | Area under active irrigation, a lead on metal output | Heap-leach gold and copper miners | High | Medium (method unproven) |
| 7 | City open-data police crash records | Auto claim frequency before insurers report | PGR, ALL and other personal auto writers | Low | Medium-high |

## Proposed sources in detail

Each idea lists the public data, a first build, the ground truth to validate against, what the novelty search turned up, and the traps.

### 1. ER wait times as a hospital-demand nowcast

- **Data**: HCA facilities post average ER wait times in real time on their websites, billboards and text lines ([HCA ED Wait Time](https://hcaefddoc.com/applications/ed-wait-time.dot)). The figure is time to see a clinician, a four-hour rolling average refreshed every 30 minutes ([HealthONE](https://www.healthonecares.com/specialties/emergency-care)). Other systems such as AdventHealth and Orlando Health post similar feeds ([CityDesk Orlando](https://citydeskorlando.com/health-wellness/hospital-er-wait-times-orlando-comparison/)).
- **Build**: poll each facility every 30 minutes, build a market-level congestion index, and compare HCA markets against non-profit neighbours.
- **Validate**: reported ER visits and admissions each quarter; Tenet, for example, reported ER visits up 2% in Q2 2026 ([Globe and Mail](https://theglobeandmail.com/investing/markets/stocks/HCA/pressreleases/4835107/can-shorter-hospital-stays-help-hca-handle-more-patient-demand)).
- **Novelty check**: only consumer aggregators republish these times ([ercost.com](https://www.ercost.com/cities/houston/)); no investment use found.
- **Traps**: wait time measures congestion, not volume, and moves with staffing. Feeds can be switched off, as Health PEI did during COVID ([CBC](https://www.cbc.ca/lite/story/1.6002843)). Check site terms before scraping.

### 2. Seismic blast catalog as a quarry-activity tracker

- **Data**: the USGS ComCat catalog labels events as quarry blasts, with time, location and magnitude ([example event](https://earthquake.usgs.gov/earthquakes/eventpage/mb90116173)). One recent Georgia event was reclassified as a blast near a Vulcan Materials quarry ([CBS Atlanta](https://www.cbsnews.com/atlanta/news/usgs-reclassifies-2-3-magnitude-georgia-earthquake-as-quarry-blast/)).
- **Build**: pull blast-type events through the catalog API, geofence them to known quarries and mines, and count blasts per site per week.
- **Validate**: USGS quarterly aggregates estimates, which lag the quarter; Q2 2026 output was 674 million tonnes, up 5.7% ([Pit & Quarry](https://www.pitandquarry.com/usgs-aggregate-production-climbs-in-second-quarter/)). Also company shipment volumes.
- **Novelty check**: mining seismicity research targets safety, not output; no economic use of blast counts found.
- **Traps**: labelling depends on regional networks, so coverage is uneven by state. Blast count is not tonnage. Stretch goal: detect smaller blasts in raw waveforms from public citizen seismometers ([Raspberry Shake station map](https://www.seismo.ethz.ch/export/sites/sedsite/knowledge/.galleries/pdf_for-schools/Earthquake-Monitoring_Raspberry-Shake_June-2025_Solutions.pdf_2063069299.pdf)).

### 3. Bagasse power exports as a Brazilian cane-crush nowcast

- **Data**: Brazil's grid operator publishes verified hourly generation for each plant, in monthly files through September 2026 ([ONS open data](https://dados.ons.org.br/dataset/geracao-usina-2)), also on a public S3 bucket ([AWS registry](https://registry.opendata.aws/ons-opendata-portal/)). Mills burn bagasse, the crushing residue, and export surplus power to the grid ([UNICA bioelectricity report](https://unicadata.com.br/arquivos/pdfs/2024/07/70094aec82c0fafeaae7a5d6d7982be9.pdf)).
- **Why now**: UNICA's fortnightly crush reports went silent in late June 2026, leaving a two-month data gap ([Bloomberg](https://www.bloomberg.com/news/articles/2026-08-06/world-sugar-market-s-most-trusted-dataset-becomes-more-opaque)).
- **Build**: tag bagasse-fired plants, sum daily Center-South output, and regress past UNICA fortnightly crush on it.
- **Novelty check**: satellite harvest tracking is taken (see above); UNICA uses grid data only for monthly energy-sector reporting. No crush nowcast from plant-level generation found.
- **Traps**: mills consume much of their own power, and exports respond to spot power prices. Small plants in this dataset carry forecasts, not measurements ([data dictionary](https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/geracao_usina_2_ho/DicionarioDados_GeracaoPorUsina.pdf)).

### 4. FAA obstruction filings as a construction pipeline

- **Data**: anything over 200 ft, or near an airport, needs FAA notice at least 45 days before construction, including temporary cranes ([San Diego bulletin](https://www.sandiego.gov/development-services/forms-publications/information-bulletins/520), [FAA OEG briefing](https://www.faa.gov/air_traffic/flight_info/aeronav/acf/media/Briefings/OEG.pdf)). The FAA processes over 160,000 such studies a year, and case data can be downloaded as CSV ([USFWS method note](https://www.fws.gov/sites/default/files/documents/General_Information_and_Metadata_20231114.pdf)).
- **Build**: weekly counts of new filings by structure type (crane, tower, building) and metro; use the follow-up notice of actual construction as the start signal.
- **Validate**: Census construction spending by region; tower REIT new-build disclosures.
- **Novelty check**: the wind-turbine slice already feeds the US Wind Turbine Database, updated from weekly FAA files ([Scientific Data](https://www.nature.com/articles/s41597-020-0353-6)). No investment use of the non-wind filings found.
- **Traps**: many proposals are never built, and one project can file many cases.

### 5. Offshore helicopter tracks as a platform-activity signal

- **Data**: public ADS-B networks record helicopter flights. The Gulf of Mexico fleet was put at 650+ helicopters serving about 5,000 platforms, with ground stations placed on rigs (2007 figures, [Aviation Today](https://www.aviationtoday.com/2007/04/01/ads-b-in-the-gulf/)).
- **Build**: match flight endpoints to platform helidecks, count landings per platform per week, and flag spikes (drilling, turnarounds) and drops (shut-ins).
- **Validate**: monthly lease-level production from US offshore regulators (lagged); operator quarterly reports.
- **Novelty check**: a 2025 paper uses vessel and helicopter tracks for offshore wind maintenance ([IOP](https://iopscience.iop.org/article/10.1088/1742-6596/3025/1/012016)); no oil and gas trading use found.
- **Traps**: low-altitude coverage gaps offshore, weather-driven noise, medevac and rescue flights mixed in.

### 6. Heap-leach pad wetting as a metal-output lead

- **Data**: free Sentinel-2 imagery; its shortwave-infrared bands respond to moisture ([Springer study](https://link.springer.com/article/10.1007/s12145-025-01926-6)). A global dataset already outlines heap-leach pads as polygons ([Nature Comms Earth & Env](https://www.nature.com/articles/s43247-023-00805-6)).
- **Build**: inside each pad, classify wet versus dry pixels and track irrigated area over time.
- **Validate**: quarterly ounces or tonnes from single-asset producers.
- **Novelty check**: pad mapping is static; no production nowcast found. The closest analogue, lithium pond area versus output, is already published (see rejected list), which suggests the approach can work.
- **Traps**: the method is unproven. Drip lines under cover may be invisible, and leach cycles lag output by months.

### 7. Police crash records as an auto-claims frequency nowcast

- **Data**: large city open-data portals publish police-reported crashes with a short lag (NYC and Chicago are the usual starting points; confirm current lag and fields before the session). Industry frequency benchmarks come from insurer-contributed data that is not public ([CCC Crash Course](https://www.cccis.com/reports/crash-course-2025/q2)).
- **Build**: crashes per vehicle-mile by metro and month, normalised with federal traffic-volume data.
- **Validate**: Progressive's monthly results give an unusually fast target (general knowledge; confirm the current disclosure format).
- **Novelty check**: no trading use of public crash feeds found.
- **Traps**: police reports undercount minor damage claims, reporting rules change, and city footprints differ from insurers' books.

## Rejected after the novelty check

Each of these looked fresh at first and failed the screen; they make good live examples of why the check matters.

| Candidate | Why it failed | Evidence |
| --- | --- | --- |
| Ski-lift queue times for Vail skier visits | Same question already answered by geolocation panels | [TownLift on Placer.ai](https://townlift.com/2022/05/customer-cell-phone-data-indicates-record-visitation-at-park-city-mountain-for-the-21-22-ski-season/) |
| Lithium evaporation-pond area for Atacama output | Published model links pond area to production | [MDPI Sustainability](https://www.mdpi.com/2071-1050/17/12/5631) |
| EU charger status feeds (mandatory since April 2025) for charging operators | Utilisation products already built on them | [AFIR guide](https://www.ampeco.com/guides/afir-compliance-guide-charge-point-operators/), [Chargalytics](https://www.chargalytics.com/) |
| Data center generator air permits for AI capex | Public permit trackers exist | [SAVRN tracker](https://savrn.com/blog/data-center-permit-tracker), [Better Data Center Project](https://betterdatacenterproject.com/wp-content/uploads/2026/03/Diesel-Generators-at-Data-Centers-Status-Impacts-and-Protective-Practices.pdf) |
| Landsat heat islands for data center load | Journalists and commercial thermal satellites cover it | [Pulitzer Center](https://pulitzercenter.org/stories/data-centers-heat-behind-cloud), [ESA on HotSat-1](https://earth.esa.int/eogateway/missions/hotsat-1) |
| npm and PyPI downloads for dev-tool stocks | Packaged trackers pitched to investors; PyPI counts also break on 2026-08-24 | [Apify tracker](https://apify.com/scrapemint/package-adoption-tracker), [PyPI blog](https://blog.pypi.org/posts/2026-08-31-download-counts/) |
| Cinema seat maps for AMC and Cinemark attendance | Commercial seat-occupancy scraping offered | [Actowiz](https://www.actowizsolutions.com/amc-theatre-market-intelligence-usa.php) |
| Satellite cane harvest progress during the UNICA gap | Harvest-monitoring services and papers exist | [EOSDA](https://eos.com/blog/sugarcane-harvest-monitoring-in-brazil-dates-and-areas/) |

## Suggested workshop format

Run it as a team exercise on the shortlist, one source per team, with the novelty check as the first deliverable.

1. **Landscape (30 min)**: walk through the "already taken" table and one rejected case, then teach the check: data marketplaces, scraper marketplaces, Google Scholar, GitHub.
2. **Re-check novelty (30 min)**: each team confirms its source is still unclaimed and writes down the closest adjacent work.
3. **Acquire (90 min)**: pull history. Sources 2, 3 and 4 have downloadable archives; source 7 usually does too.
4. **Build and validate (2 h)**: construct the signal, align it to the ground truth named above, and test out of sample.
5. **Readout (45 min)**: lead-lag result, biggest trap found, and what it would take to productionise.

Two practical notes. ER wait times (source 1) have no public archive, so start a collector several weeks before the session. Have compliance review scraping terms of use for sources 1 and 5 in advance.

## Sources

Findings are from web search results reviewed on the as-of date; links are also given inline above.

- [HCA Healthcare: ED Wait Time application](https://hcaefddoc.com/applications/ed-wait-time.dot)
- [HealthONE: ER wait time definition](https://www.healthonecares.com/specialties/emergency-care)
- [USGS ComCat quarry blast event](https://earthquake.usgs.gov/earthquakes/eventpage/mb90116173)
- [Pit & Quarry: USGS Q2 aggregates](https://www.pitandquarry.com/usgs-aggregate-production-climbs-in-second-quarter/)
- [ONS: hourly generation by plant](https://dados.ons.org.br/dataset/geracao-usina-2)
- [Bloomberg: UNICA data gap](https://www.bloomberg.com/news/articles/2026-08-06/world-sugar-market-s-most-trusted-dataset-becomes-more-opaque)
- [FAA Obstruction Evaluation Group briefing](https://www.faa.gov/air_traffic/flight_info/aeronav/acf/media/Briefings/OEG.pdf)
- [US Wind Turbine Database paper](https://www.nature.com/articles/s41597-020-0353-6)
- [IOP: offshore wind O&M from AIS and ADS-B](https://iopscience.iop.org/article/10.1088/1742-6596/3025/1/012016)
- [Global mining footprint dataset](https://www.nature.com/articles/s43247-023-00805-6)
- [MDPI: lithium pond area and output](https://www.mdpi.com/2071-1050/17/12/5631)
- [Chargalytics](https://www.chargalytics.com/)
- [SAVRN data center permit tracker](https://savrn.com/blog/data-center-permit-tracker)
- [PyPI blog: download count change](https://blog.pypi.org/posts/2026-08-31-download-counts/)
