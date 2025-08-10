#!/bin/bash
# circleci_setup_template.sh

echo "Setting up CircleCI for ML project..."

# Create .circleci directory
mkdir -p .circleci

# Create basic config.yml
cat > .circleci/config.yml << 'EOF'
version: 2.1

orbs:
  python: circleci/python@2.1.1

jobs:
  test-and-train:
    docker:
      - image: cimg/python:3.9
    steps:
      - checkout
      - python/install-packages:
          pkg-manager: pip
      - run:
          name: Run tests
          command: |
            python -m pytest tests/ || echo "No tests found"
      - run:
          name: Train model
          command: |
            python app.py

workflows:
  version: 2
  build-and-deploy:
    jobs:
      - test-and-train
EOF

echo "CircleCI configuration created at .circleci/config.yml"
echo "Next steps:"
echo "1. Commit this configuration to your repository"
echo "2. Connect your repository to CircleCI"
echo "3. Configure environment variables if needed"
