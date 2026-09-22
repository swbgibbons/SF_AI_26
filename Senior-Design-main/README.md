# Senior-Design 2258-CSE-4316-005

A funded project in Senior Design class with State Farm for ML/AL application

# Team member

- Thanh Tran
- Kasson Davis
- Miguel Oropeza
- Jeffery Aguirre
- Serah Jolly
- Miwa Yoshida

# Project Details

-Ver 0: waiting for details

# Website Instructions for cloning the backend MAKE SURE TIME ZONE SYNC, DEPENDANCIES INSTALL, DONT USE PYTHON MICROSOFT INSTALL!

1. `pip install uv`
2. `cd backend` first, because `manage.py` is inside this folder.
3. `pip sync`
4. Put the `.env` and `key.json` files in the `backend` folder.
5. Run `uv run python manage.py runserver` while you are still inside `backend`.
6. The frontend also needs to be running. `localhost:8000/api/login` is the backend login page and it redirects to `localhost:8080`.

# For contributing users

1. git fetch
2. git reset --hard origin/react
