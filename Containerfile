FROM registry.redhat.io/ubi9/python-312:latest
USER root
WORKDIR /app
COPY . /app
RUN chgrp -R 0 /app && \
    chmod -R g=u /app
RUN pip install -r requirements.txt
USER 1001
ENV HOST=0.0.0.0
ENV PORT=7861
CMD ["python", "agent-triage-analysis.py"]