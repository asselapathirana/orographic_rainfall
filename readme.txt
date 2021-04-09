Sometimes dokku can't find python version heroku supports
first check : https://devcenter.heroku.com/articles/python-support#supported-runtimes (usually the dokku error will indicate the heroku version using)
and create/edit runtime.txt to match a supported version

then add a file .env with 
export BUILDPACK_URL=https://github.com/heroku/heroku-buildpack-python


