# For finding latest versions of the base image see
# https://github.com/SwissDataScienceCenter/renkulab-docker
ARG RENKU_BASE_IMAGE=renku/renkulab-py:3.7-0.7.3
FROM ${RENKU_BASE_IMAGE}

# Uncomment and adapt if code is to be included in the image
# COPY src /code/src

# Uncomment and adapt if your R or python packages require extra linux (ubuntu) software
# e.g. the following installs apt-utils and vim; each pkg on its own line, all lines
# except for the last end with backslash '\' to continue the RUN line
#
# USER root
# RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    apt-utils \
#    vim
# USER ${NB_USER}

# install the python dependencies
COPY Pipfile Pipfile.lock /tmp/
RUN pip install pipenv
RUN cd /tmp/ & pipenv install --system --deploy --ignore-pipfile

# RENKU_VERSION determines the version of the renku CLI
# that will be used in this image. To find the latest version,
# visit https://pypi.org/project/renku/#history.
ARG RENKU_VERSION=0.12.2

########################################################
# Do not edit this section and do not add anything below

RUN if [ -n "$RENKU_VERSION" ] ; then \
    pipx uninstall renku && \
    pipx install --force renku==${RENKU_VERSION} \
    ; fi

########################################################