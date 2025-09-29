# Contributing to the Project

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## We Develop with Github
We use github to host code, to track issues and feature requests, as well as accept pull requests.

## We Use [Github Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `master`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/mohin-io/levy-model/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](); it's that easy!

## Write bug reports with detail, background, and sample code

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can.
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Use a Consistent Coding Style

* 2 spaces for indentation rather than tabs
* You can try running `npm run lint` for style unification

## Branching and Commit Conventions

To maintain a clean and understandable history, please follow these conventions:

### Branch Naming

*   **Feature branches:** `feature/<short-description>` (e.g., `feature/add-cgmy-model`)
*   **Bugfix branches:** `bugfix/<issue-number>-<short-description>` (e.g., `bugfix/123-fix-api-error`)
*   **Hotfix branches:** `hotfix/<short-description>` (e.g., `hotfix/critical-api-patch`)
*   **Release branches:** `release/<version>` (e.g., `release/v1.0.0`)

### Commit Messages

Commit messages should be clear, concise, and follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps with automated changelog generation and understanding the purpose of each commit.

**Format:** `<type>(<scope>): <description>`

*   **type:** `feat` (new feature), `fix` (bug fix), `docs` (documentation only changes), `style` (code style, formatting), `refactor` (code refactoring), `perf` (performance improvement), `test` (adding missing tests), `build` (changes that affect the build system), `ci` (CI related changes), `chore` (other changes that don't modify src or test files).
*   **scope (optional):** The part of the codebase affected (e.g., `api`, `pricing-engine`, `docs`, `calibration-net`).
*   **description:** A short, imperative, present tense description of the change.

**Examples:**
*   `feat(pricing-engine): add CGMY model characteristic function`
*   `fix(api): handle missing option_prices in request`
*   `docs: update README with project structure`
*   `chore(deps): update tensorflow to latest version`

### Release Tagging

Releases will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) (MAJOR.MINOR.PATCH). Tags will be prefixed with `v` (e.g., `v1.0.0`).

## License
By contributing, you agree that your contributions will be licensed under its MIT License.
