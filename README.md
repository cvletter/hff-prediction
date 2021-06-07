# Productie voorspelmodel
Versie 1.0 | 7 juni 2021

## Installatie

Om dit model te installeren, onderneem onderstaande stappen. Dit stappenplan gaat er vanuit dat in omgeving beschikbaar zijn: 
- Git
- Python 3.8.5 (of hoger)
- PowerShell

Stappen:
1. Maak een map op een schijf naar keuze, bijv. de map met naam: `Productie Voorspelmodel`, op schijf `E:`.
2. De basis waar alles zal worden opgeslagen is onder: `E:\Productie Voorspelmodel`.
3. Zorg voor een github account en `git` installatie.
4. Navigeer middels de PowerShell - Run deze als _Administrator_ - naar de map onder 2, doe dit met het volgende commando: `cd E:\Productie Voorspelmodel`
5. Voer nu het commando `git clone https://github.com/cvletter/hff-prediction.git` uit om de package te installeren.
6. Navigeer vervolgens naar de map `hff-prediction`, door het commando `cd hff-prediction` uit te voeren.
7. Installeer nu het model met onderstaande 2 commando's:

Installatie van het voorspelmodel
```bash
pip install -r requirements.txt
pip install .
```

Installatie van development / testingmodule
```bash
pip install -r requirements-dev.txt
pip install -Ue .
```



## Data input en output
De volgende stappen omschrijven het inladen en opslaan van de benodigde data om het model te draaien. 

Voordat data kan worden opgeslagen moet eerst de juiste structuur worden opgezet. Deze kan worden gemaakt door het volgende commando uit te voeren: `hff --mode setup_structure --output "E:"`. Vervang hier `E:` voor de schijf waar `Productie Voorspelmodel`staat. Dit genereert onderstaande structuur.

##### Folder structuur
```
E:
|- Productie Voorspelmodel
    |- Input
    |    |- Bestellingen
    |    |- Campagnes
    |    |- Productstatus
    |- Processed
    |    |- Bestellingen
    |    |- Campagnes
    |    |- Features
    |    |- Superunie
    |    |- Weer
    |- Output
    |    |- Testresultaten
    |    |- Tussenvoorspellingen
    |    |- Voorspellingen

```

Stappen:
1. In de map `Input` wordt de ruwe data opgeslagen, het wekelijkse bestand met bestellingen moet worden opgeslagen in de submap `Input\Bestellingen`.
    1. De data dient te worden opgeslagen als `.csv` bestand.
    2. Zie onderstaand voor de benodigde kolommen.
    3. De datum range loopt van `1 augustus 2018` tot en met de laatst beschikbare datum.

2. In de map `Campagnes` staat een overzicht van campagnes die zijn geweest of gepland voor de producten. Deze dient per kwartaal te worden geupdatet.
3. De voorspellingen verschijnen in de `Output` map onder submap `Voorspellingen`
    1. De voorspellingen hebben naamstructuur: `predictions_p{w}_d{prediction date}_{date}_{hour}.csv`
        1. `w`: predictie window, deze staat standaard op `2`.
        2. `prediction_date`: Week waarvoor voorspelling wordt gemaakt.
        3. `date` & `hour`: Datum en tijdstip waarop bestand is aangemaakt.
4. Bestanden worden automatisch verwijderd als het model opnieuw wordt gedraaid. Het bewaard enkel het laatste bestand en het voorlaatste bestand. Dit om te voorkomen dat schijf volloopt.


##### Data input kolommen
```
"Artikelgroep"
"ConsGrp Naam"
"Organisatie"
"Superunie"
"InkoopRecept"
"InkoopRecept Omschrijving"
"Weekjaar"
"Periode"
"Week"
"Leverdatum
"Order"
"Artikelen"
"Artikelomschrijving"
"Besteld #CE"
```

## Voorspelling maken
Om een voorspelling te maken moeten de volgende stappen worden uitgevoerd. Het model maakt nu een voorspelling voor een _hele week_ en gebruikt altijd de datum van de maandag om een voorspelling voor een week aan te geven. Bijvoorbeeld `2021-05-31` geeft de totaalvoorspelling weer voor de week van 31 mei 2021.

Stappen:
1. Het volgende basiscommand moet worden uitgevoerd om een voorspelling te realiseren: `hff --mode predict --reload`.
    1. `--predict` vertelt dat het de voorspelmodule moet activeren.
    2. `--reload` zorgt ervoor dat de nieuwste data wordt ingeladen.
2. Additionele parameters zijn beschikbaar, maar staan nu automatisch ingesteld, deze zijn:
    1. `--window` staat nu op `2`, dit zorgt voor een voorspelling 2 weken vooruit.
    2. `--date` staat nu op automatische detectie, het toevoegen van deze parameter plus een datum volgens de structuur `yyyy-mm-dd`, zorgt voor een voorspelling van die datum. Bijvoorbeeld: `hff --mode predict --date "2021-05-31" --window 2 --reload`. 


##### Logging
Het proces van het voorspelmodel wordt gelogd in bestand `log_productievoorspelmodel.log`. Dit bestand is te vinden in de map van het model: `hff-prediction`. Bij een foutmelding kan in dit bestand worden gekeken voor de reden.

 