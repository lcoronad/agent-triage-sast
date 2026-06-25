FROM registry.redhat.io/ubi9/ubi:9.8-1782121068
USER root
WORKDIR /workspace
RUN chgrp -R 0 /workspace && \
    chmod -R g=u /workspace && \
    dnf install -y jq && \
    dnf install -y maven && \
    dnf clean all
USER 1001
CMD ["--help"]