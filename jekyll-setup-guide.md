# GitHub Pages Setup Guide for Climate Smart Indonesia

This guide will help you set up your repository as a GitHub Pages site to display your Climate Smart Indonesia documentation.

## Steps to Set Up GitHub Pages

1. **Use Your Existing Repository**
   - Your repository is named `Climate-Smart-Indonesia---Documentation`
   - The URL will be `normanmul.github.io/Climate-Smart-Indonesia---Documentation`

2. **Add the Documentation Files to Your Repository**
   Make sure your repository contains these files in the root directory:
   - `_config.yml` - Configuration for GitHub Pages
   - `index.md` - The main landing page
   - `404.html` - Custom 404 page for when users navigate to a non-existent page
   - `Gemfile` - Specifies the Ruby gems needed (optional)
   - Your documentation markdown files:
     - `dengue-prediction-documentation.md`
     - `model-api-docs.md`
     - `code-examples.md`
     - `deployment-guide.md`

3. **File Structure**
   Your repository structure should look like:
   ```
   Climate-Smart-Indonesia---Documentation/
   ├── _config.yml
   ├── index.md
   ├── 404.html
   ├── Gemfile (optional)
   ├── dengue-prediction-documentation.md
   ├── model-api-docs.md
   ├── code-examples.md
   └── deployment-guide.md
   ```

4. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Click "Settings"
   - Scroll down to the "GitHub Pages" section
   - Under "Source", select "main branch" (or your default branch)
   - Click "Save"
   - Wait a few minutes for your site to build

5. **Access Your Site**
   - Your site will be available at: `https://normanmul.github.io/Climate-Smart-Indonesia---Documentation/`

## Customizing Your Site

### Themes

You can change the theme in your `_config.yml` file. Some popular themes:
- minima
- cayman (currently selected)
- jekyll-theme-minimal
- jekyll-theme-slate
- jekyll-theme-tactile

To change the theme, update the `remote_theme` line in `_config.yml`:
```yaml
remote_theme: pages-themes/minimal@v0.2.0
```

### Custom Domain (Optional)

1. Buy a domain name from a registrar (GoDaddy, Namecheap, etc.)
2. In your repository settings, under GitHub Pages, enter your custom domain
3. Add a CNAME record at your domain registrar pointing to `normanmul.github.io`
4. Add a file named `CNAME` to your repository with your domain name

## Troubleshooting

- **Site Not Building**: Check the "GitHub Pages" section in your repository settings for error messages
- **Styling Issues**: Make sure your theme is correctly specified in `_config.yml`
- **Broken Links**: Ensure all links between pages use the `.html` extension, not `.md`
- **Custom Domain Not Working**: DNS changes can take up to 48 hours to propagate

## GitHub Actions for Automated Builds (Advanced)

For more control over the build process, you can use GitHub Actions:

1. Create a `.github/workflows/github-pages.yml` file:

```yaml
name: Build and deploy Jekyll site to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  github-pages:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: helaili/jekyll-action@v2
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
```

2. This will automatically build and deploy your site when you push to the main branch

## Local Testing (Optional)

To test your site locally before pushing:

1. Install Ruby and Jekyll
   ```
   gem install bundler jekyll
   ```

2. Run the local server
   ```
   bundle exec jekyll serve
   ```

3. View your site at `http://localhost:4000`

## Credits

Documentation site created by:
- Naufal Prawironegoro
- Dr. Kaushik Sharkar
