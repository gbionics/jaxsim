# Contributing to JAXsim :rocket:

Hello Contributor,

We're thrilled that you're considering contributing to JAXsim!
Here's a brief guide to help you seamlessly become a part of our project.

## Development Environment :hammer_and_wrench:

Make sure your development environment is set up.
Follow the installation instructions in the [README](./README.md) to get JAXsim and its dependencies up and running.

To ensure consistency and maintain code quality, we recommend using the pre-commit hook with the following configuration.
This will help catch issues before they become a part of the project.

### Setting Up Pre-commit Hook :fishing_pole_and_fish:

`pre-commit` is a tool that manages pre-commit hooks for your project.
It will run checks on your code before you commit it, ensuring that it meets the project's standards.

First, install `pre-commit` if you haven't already:

```bash
pip install pre-commit
```

Then, run the following command to install the hooks:

```bash
pre-commit install
```

### Using Pre-commit Hook :vertical_traffic_light:

Before making any commits, the pre-commit hook will automatically run.
If it finds any issues, it will prevent the commit and provide instructions on how to fix them.

To get your commit through without fixing the issues, use the `--no-verify` flag:

```bash
git commit -m "Your commit message" --no-verify
```

To manually run the pre-commit hook at any time, use:

```bash
pre-commit run --all-files
```

## Making Changes :construction:

Before submitting a pull request, create an issue to discuss your changes if major changes are involved.
This helps us understand your needs and provide feedback.
Clearly describe your pull request, referencing any related issues.
Follow the [PEP 8](https://peps.python.org/pep-0008/) style guide and include relevant tests.

## Testing :test_tube:

Your code will be tested with the CI/CD pipeline before merging.
Feel free to add new ones or update the existing tests in the [workflows](./.github/workflows) folder to cover your changes.

## Documentation :book:

Update the documentation in the [docs](./docs) folder and the [README](./README.md) to reflect your changes, if necessary.
There is no need to build the documentation locally; it will be automatically built and deployed with your pull request, where a preview link will be provided.

## Code Review :eyes:

Expect feedback during the code review process.
Address comments and make necessary changes.
This collaboration ensures quality.
Please keep the commit history clean, or squash commits if necessary.

## License :scroll:

JAXsim is under the [BSD 3-Clause License](./LICENSE).
By contributing, you agree to the same license.

Thank you for contributing to JAXsim! Your efforts are appreciated.
