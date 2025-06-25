# 🚀 Deployment Guide for Spam Email Classifier

This guide covers multiple hosting options for your Spam Email Classifier web application.

## 📋 Prerequisites

Before deploying, ensure you have:
- Trained models in the `models/` directory
- All dependencies listed in `requirements.txt`
- Your application tested locally

## 🔧 Hosting Options

### 1. 🟢 Heroku (Recommended for beginners)

**Pros:** Easy setup, automatic scaling, good free tier
**Cons:** Can be slow on free tier, limited by dyno hours

#### Steps:
```bash
# 1. Install Heroku CLI
# Visit: https://devcenter.heroku.com/articles/heroku-cli

# 2. Login to Heroku
heroku login

# 3. Create a new Heroku app
heroku create your-spam-classifier-app

# 4. Set environment variables
heroku config:set FLASK_ENV=production

# 5. Deploy
git add .
git commit -m "Prepare for Heroku deployment"
git push heroku main

# 6. Scale your app
heroku ps:scale web=1

# 7. Open your app
heroku open
```

#### Required files (already created):
- `Procfile` - Tells Heroku how to run your app
- `runtime.txt` - Specifies Python version
- Modified `app.py` - Handles PORT environment variable

### 2. 🔵 Railway (Modern alternative to Heroku)

**Pros:** Fast deployment, generous free tier, modern interface
**Cons:** Newer platform, smaller community

#### Steps:
```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login
railway login

# 3. Initialize project
railway init

# 4. Deploy
railway up

# 5. Set custom domain (optional)
railway domain
```

#### Configuration:
- `railway.toml` file already created
- Automatic health checks on `/health` endpoint

### 3. 🟣 Vercel (Serverless)

**Pros:** Fast global CDN, excellent for static sites with API
**Cons:** Cold starts, limited to serverless functions

#### Steps:
```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Deploy
vercel

# 3. Follow prompts to configure
```

#### Configuration:
- `vercel.json` file already created
- Note: Large ML models may exceed size limits

### 4. 🐳 Docker + Any Cloud Provider

**Pros:** Consistent environment, works everywhere
**Cons:** Requires container knowledge

#### Steps:
```bash
# 1. Build Docker image
docker build -t spam-classifier .

# 2. Test locally
docker run -p 5000:5000 spam-classifier

# 3. Deploy to your preferred cloud:
# - AWS ECS/Fargate
# - Google Cloud Run
# - Azure Container Instances
# - DigitalOcean App Platform
```

#### For Docker Compose (local development):
```bash
docker-compose up --build
```

### 5. ☁️ Cloud Platforms

#### AWS (EC2 + Elastic Beanstalk)
```bash
# 1. Install EB CLI
pip install awsebcli

# 2. Initialize
eb init

# 3. Create environment
eb create production

# 4. Deploy
eb deploy
```

#### Google Cloud Platform (App Engine)
```bash
# 1. Create app.yaml
# 2. Deploy
gcloud app deploy
```

#### Azure (App Service)
```bash
# 1. Create resource group
az group create --name spam-classifier-rg --location "East US"

# 2. Create app service plan
az appservice plan create --name spam-classifier-plan --resource-group spam-classifier-rg --sku B1 --is-linux

# 3. Create web app
az webapp create --resource-group spam-classifier-rg --plan spam-classifier-plan --name your-spam-classifier --runtime "PYTHON|3.9"

# 4. Deploy
az webapp up --name your-spam-classifier
```

## 🔒 Security Considerations

### Environment Variables
Set these in your hosting platform:
```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

### HTTPS
Most platforms provide HTTPS automatically. For custom domains:
- Heroku: Automatic with custom domains
- Railway: Automatic
- Vercel: Automatic

## 📊 Performance Optimization

### Model Loading
For better performance, consider:
1. **Model caching** - Load models once at startup
2. **Lighter models** - Use quantized or distilled models
3. **Model serving** - Use TensorFlow Serving or TorchServe

### Memory Management
```python
# Add to app.py for better memory usage
import gc
import torch

# After model loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
```

## 🔍 Monitoring

### Health Checks
Your app includes a `/health` endpoint for monitoring:
```bash
curl https://your-app.herokuapp.com/health
```

### Logs
- **Heroku:** `heroku logs --tail`
- **Railway:** `railway logs`
- **Vercel:** Check dashboard

## 💰 Cost Estimates

### Free Tiers:
- **Heroku:** 550-1000 dyno hours/month
- **Railway:** 500 hours + $5 credit
- **Vercel:** 100GB bandwidth, 10GB storage
- **Google Cloud:** $300 credit for 90 days

### Paid Options:
- **Heroku Hobby:** $7/month
- **Railway Pro:** $5/month
- **AWS/GCP/Azure:** ~$10-50/month depending on usage

## 🚀 Quick Start Commands

### For Heroku:
```bash
git clone <your-repo>
cd spam-classifier
heroku create your-app-name
git push heroku main
```

### For Railway:
```bash
git clone <your-repo>
cd spam-classifier
railway init
railway up
```

### For Docker:
```bash
git clone <your-repo>
cd spam-classifier
docker build -t spam-classifier .
docker run -p 5000:5000 spam-classifier
```

## 🆘 Troubleshooting

### Common Issues:
1. **Model files too large:** Use Git LFS or model compression
2. **Memory errors:** Reduce model size or upgrade hosting plan
3. **Slow cold starts:** Keep your app warm with monitoring services
4. **Import errors:** Ensure all dependencies in requirements.txt

### Debug Commands:
```bash
# Check app status
heroku ps

# View logs
heroku logs --tail

# Restart app
heroku restart
```

## 📞 Support

If you encounter issues:
1. Check the logs first
2. Verify all environment variables are set
3. Test locally with production settings
4. Consult platform-specific documentation

---

Choose the hosting option that best fits your needs, budget, and technical requirements!
