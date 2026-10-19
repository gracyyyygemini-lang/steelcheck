# data/

Drop the **AISC Shapes Database** here as `aisc_shapes.csv` to get full shape
coverage. AISC publishes it free alongside the Steel Construction Manual.

Nothing in this folder is tracked by git (see `.gitignore`) — the database has
its own license terms and is not redistributed with this project. Check them
before copying the file anywhere.

    import steelcheck as sc
    sc.shapes.load_aisc_csv("data/aisc_shapes.csv")
