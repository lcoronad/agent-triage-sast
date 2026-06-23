FROM registry.redhat.io/ubi9/ubi:9.8-1782121068
USER root
WORKDIR /workspace
RUN chgrp -R 0 /workspace && \
    chmod -R g=u /workspace
RUN curl -fsSL https://raw.githubusercontent.com/opengrep/opengrep/main/install.sh | bash
USER 1001
ENTRYPOINT ["opengrep"]
CMD ["--help"]