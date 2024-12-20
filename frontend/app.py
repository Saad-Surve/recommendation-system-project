from fastapi import FastAPI, Request, Form, HTTPException, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pathlib import Path
import httpx
import json
from typing import Optional, List
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import asyncio
from datetime import datetime

app = FastAPI(
    title="Article Recommender API",
    description="REST API for article recommendations and user interactions",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8001"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup template and static directories
templates = Jinja2Templates(directory="frontend/templates")
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")

API_URL = "http://localhost:8000"

@app.get("/")
async def home(request: Request):
    token = request.cookies.get("access_token")
    if token:
        return RedirectResponse(url="/dashboard")
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup")
async def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...)
):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Print debug information
            print(f"Attempting login for user: {username}")
            
            response = await client.post(
                f"{API_URL}/login",
                json={
                    "username": username,
                    "password": password
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }
            )
            
            # Print response for debugging
            print(f"Login response status: {response.status_code}")
            print(f"Login response body: {response.text}")
            
            if response.status_code == 200:
                data = response.json()
                redirect = RedirectResponse(url="/dashboard", status_code=303)
                redirect.set_cookie(
                    key="access_token",
                    value=data["access_token"],
                    httponly=True,
                    max_age=1800
                )
                return redirect
            else:
                error_msg = "Login failed. Please check your credentials."
                if response.status_code != 404:  # If we got a response
                    try:
                        data = response.json()
                        error_msg = data.get("detail", error_msg)
                    except:
                        pass
                return templates.TemplateResponse(
                    "login.html",
                    {
                        "request": request,
                        "error": error_msg
                    }
                )
        except Exception as e:
            print(f"Login error: {str(e)}")  # Debug print
            return templates.TemplateResponse(
                "login.html",
                {
                    "request": request,
                    "error": f"Server error: {str(e)}"
                }
            )

@app.post("/signup")
async def signup(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{API_URL}/signup",
                json={
                    "username": username,
                    "email": email,
                    "password": password
                }
            )
            if response.status_code == 200:
                data = response.json()
                response = RedirectResponse(url="/select-interests", status_code=303)
                response.set_cookie(
                    key="access_token",
                    value=data["access_token"],
                    httponly=True,
                    max_age=1800
                )
                return response
        except Exception as e:
            pass

        return templates.TemplateResponse(
            "signup.html",
            {
                "request": request,
                "error": "Registration failed. Please try again."
            }
        )

@app.get("/dashboard")
async def dashboard(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient(timeout=75.0) as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                response = RedirectResponse(url="/")
                response.delete_cookie("access_token")
                return response
                
            user_data = user_response.json()
            print(f"User data: {user_data}")  # Debug print
            
            # Get user's ID from the database
            with sqlite3.connect('recommender.db') as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT user_id FROM users WHERE username = ?", 
                    (user_data['username'],)
                )
                result = cursor.fetchone()
                user_id = result[0] if result else None
                print(f"User ID: {user_id}")  # Debug print
            
            # Get personalized recommendations if we have a user_id
            recommended_articles = []
            if user_id:
                recommendations_response = await client.get(
                    f"{API_URL}/recommendations/?user_id={user_id}&n_recommendations=5",
                    headers={"Authorization": f"Bearer {token}"}
                )
                if recommendations_response.status_code == 200:
                    recommended_articles = recommendations_response.json()
                print(f"Found {len(recommended_articles)} recommended articles")  # Debug print
            
            # Get trending articles
            trending = await client.get(
                f"{API_URL}/articles/trending?n_articles=10",
                headers={"Authorization": f"Bearer {token}"}
            )
            trending_articles = []
            if trending.status_code == 200:
                trending_articles = trending.json()
            print(f"Found {len(trending_articles)} trending articles")  # Debug print
            
            # Get user stats
            stats_response = await client.get(
                f"{API_URL}/users/{user_data['username']}/interactions",
                headers={"Authorization": f"Bearer {token}"}
            )
            user_stats = {
                "views": 0,
                "likes": 0,
                "shares": 0,
                "not_interested": 0
            }
            if stats_response.status_code == 200:
                interactions = stats_response.json()
                for interaction in interactions:
                    interaction_type = interaction["interaction_type"]
                    if interaction_type == 'not_interested':
                        user_stats['not_interested'] += 1
                    else:
                        user_stats[interaction_type + "s"] += 1
            
            # Get user interests
            interests_response = await client.get(
                f"{API_URL}/users/{user_data['username']}/interests",
                headers={"Authorization": f"Bearer {token}"}
            )
            user_interests = []
            if interests_response.status_code == 200:
                user_interests = interests_response.json().get("interests", [])
            
            return templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "user": user_data,
                    "recommended_articles": recommended_articles,
                    "trending_articles": trending_articles,
                    "user_stats": user_stats,
                    "user_interests": user_interests
                }
            )
        except Exception as e:
            print(f"Dashboard error: {e}")  # Debug print
            return templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "user": user_data if 'user_data' in locals() else {"username": "User", "email": ""},
                    "recommended_articles": [],
                    "trending_articles": [],
                    "user_stats": {"views": 0, "likes": 0, "shares": 0, "not_interested": 0},
                    "user_interests": [],
                    "error": str(e)
                }
            )

@app.get("/logout")
async def logout():
    response = RedirectResponse(url="/")
    response.delete_cookie("access_token")
    return response

@app.post("/interact/{article_id}/{interaction_type}")
async def interact_with_article(
    request: Request,
    article_id: int,
    interaction_type: str
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/", status_code=303)
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info for username
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/", status_code=303)
            
            user_data = user_response.json()
            
            # Add interaction
            print(f"Adding interaction: {user_data['username']} - {article_id} - {interaction_type}")  # Debug print
            
            interaction_response = await client.post(
                f"{API_URL}/interactions/",
                json={
                    "username": user_data["username"],
                    "article_id": article_id,
                    "interaction_type": interaction_type
                },
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json"
                }
            )
            
            print(f"Interaction response: {interaction_response.status_code}")  # Debug print
            print(f"Interaction response body: {interaction_response.text}")  # Debug print
            
            if interaction_response.status_code == 200:
                if interaction_type == 'not_interested':
                    return {"status": "success", "redirect": True}
                return {"status": "success"}
            else:
                return {"status": "error", "detail": interaction_response.text}
                
        except Exception as e:
            print(f"Interaction error: {str(e)}")
            return {"status": "error", "detail": str(e)}

@app.get("/explore")
async def explore(
    request: Request,
    tag: Optional[str] = None,
    page: int = 1
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info for authentication check
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            # Get popular tags
            tags_response = await client.get(
                f"{API_URL}/articles/popular-tags",
                headers={"Authorization": f"Bearer {token}"}
            )
            popular_tags = tags_response.json() if tags_response.status_code == 200 else []
            
            # Get articles by tag if specified, otherwise get trending
            if tag:
                articles_response = await client.get(
                    f"{API_URL}/articles/by-tag/{tag}?n_articles=20",
                    headers={"Authorization": f"Bearer {token}"}
                )
            else:
                articles_response = await client.get(
                    f"{API_URL}/articles/trending?n_articles=20",
                    headers={"Authorization": f"Bearer {token}"}
                )
            
            articles = []
            if articles_response.status_code == 200:
                articles = articles_response.json()
                # Clean up tags
                for article in articles:
                    if isinstance(article['tags'], str):
                        article['tags'] = json.loads(article['tags'])
                    article['tags'] = [tag.strip().strip("[]'\"") for tag in article['tags']]
                    article['tags'] = [tag for tag in article['tags'] if tag]
            
            # Implement pagination
            items_per_page = 10
            total_items = len(articles)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            paginated_articles = articles[start_idx:end_idx]
            
            return templates.TemplateResponse(
                "explore.html",
                {
                    "request": request,
                    "articles": paginated_articles,
                    "popular_tags": popular_tags,
                    "current_tag": tag,
                    "current_page": page,
                    "total_pages": total_pages
                }
            )
        except Exception as e:
            print(f"Explore error: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.get("/search")
async def search(
    request: Request,
    q: Optional[str] = None
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Verify user is authenticated
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            # Get popular tags for initial state
            tags_response = await client.get(
                f"{API_URL}/articles/popular-tags",
                headers={"Authorization": f"Bearer {token}"}
            )
            popular_tags = tags_response.json() if tags_response.status_code == 200 else []
            
            articles = []
            if q:
                # Search articles
                search_response = await client.get(
                    f"{API_URL}/articles/search",
                    params={"query": q, "n_results": 20},
                    headers={"Authorization": f"Bearer {token}"}
                )
                if search_response.status_code == 200:
                    articles = search_response.json()
                    # Clean up tags
                    for article in articles:
                        if isinstance(article['tags'], str):
                            article['tags'] = json.loads(article['tags'])
                        article['tags'] = [tag.strip().strip("[]'\"") for tag in article['tags']]
                        article['tags'] = [tag for tag in article['tags'] if tag]
            
            return templates.TemplateResponse(
                "search.html",
                {
                    "request": request,
                    "query": q,
                    "articles": articles,
                    "popular_tags": popular_tags
                }
            )
        except Exception as e:
            print(f"Search error: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.get("/debug")
async def debug_page(request: Request):
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_URL}/debug/status")
            if response.status_code == 200:
                debug_data = response.json()
                return templates.TemplateResponse(
                    "debug.html",
                    {
                        "request": request,
                        "debug_data": debug_data
                    }
                )
        except Exception as e:
            return {"error": str(e)}

@app.get("/profile")
async def profile(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            user_data = user_response.json()
            
            # Get user interests
            interests_response = await client.get(
                f"{API_URL}/users/{user_data['username']}/interests",
                headers={"Authorization": f"Bearer {token}"}
            )
            user_interests = []
            if interests_response.status_code == 200:
                user_interests = interests_response.json().get("interests", [])
            
            # Get user stats
            stats_response = await client.get(
                f"{API_URL}/users/{user_data['username']}/interactions",
                headers={"Authorization": f"Bearer {token}"}
            )
            user_stats = {
                "views": 0,
                "likes": 0,
                "shares": 0
            }
            if stats_response.status_code == 200:
                interactions = stats_response.json()
                for interaction in interactions:
                    interaction_type = interaction["interaction_type"]
                    user_stats[interaction_type + "s"] += 1
            
            # Get recommended articles based on interests
            recommended_articles = []
            if user_interests:
                for interest in user_interests[:3]:  # Use top 3 interests
                    articles_response = await client.get(
                        f"{API_URL}/articles/by-tag/{interest}?n_articles=3",
                        headers={"Authorization": f"Bearer {token}"}
                    )
                    if articles_response.status_code == 200:
                        recommended_articles.extend(articles_response.json())
            
            return templates.TemplateResponse(
                "profile.html",
                {
                    "request": request,
                    "user": user_data,
                    "user_stats": user_stats,
                    "user_interests": user_interests,
                    "recommended_articles": recommended_articles[:6]  # Show top 6 recommendations
                }
            )
        except Exception as e:
            print(f"Profile error: {str(e)}")
            return RedirectResponse(url="/")

@app.get("/articles/{article_id}")
async def view_article(request: Request, article_id: int):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info for authentication check
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            # Get article details
            article_response = await client.get(
                f"{API_URL}/articles/{article_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if article_response.status_code != 200:
                return RedirectResponse(url="/dashboard")
                
            article = article_response.json()
            
            # Record view interaction
            await client.post(
                f"{API_URL}/interactions/",
                json={
                    "username": user_response.json()["username"],
                    "article_id": article_id,
                    "interaction_type": "view"
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            return templates.TemplateResponse(
                "article.html",
                {
                    "request": request,
                    "article": article
                }
            )
        except Exception as e:
            print(f"Article view error: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.get("/select-interests")
async def select_interests(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            # Get available tags
            tags_response = await client.get(
                f"{API_URL}/articles/popular-tags?n_tags=30",  # Get top 30 tags
                headers={"Authorization": f"Bearer {token}"}
            )
            available_tags = tags_response.json() if tags_response.status_code == 200 else []
            
            return templates.TemplateResponse(
                "select_interests.html",
                {
                    "request": request,
                    "available_tags": available_tags
                }
            )
        except Exception as e:
            print(f"Error loading interests page: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.post("/save-interests")
async def save_interests(
    request: Request,
    selected_tags: List[str] = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            user_data = user_response.json()
            
            # Save user interests
            interests_response = await client.post(
                f"{API_URL}/users/interests",
                json={
                    "username": user_data["username"],
                    "interests": selected_tags
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if interests_response.status_code == 200:
                return RedirectResponse(url="/profile", status_code=303)
            else:
                return RedirectResponse(url="/select-interests", status_code=303)
                
        except Exception as e:
            print(f"Save interests error: {str(e)}")
            return RedirectResponse(url="/select-interests")

@app.get("/admin")
async def admin_page(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            # Get current boosts
            boosts_response = await client.get(
                f"{API_URL}/articles/boosts",
                headers={"Authorization": f"Bearer {token}"}
            )
            current_boosts = []
            if boosts_response.status_code == 200:
                current_boosts = boosts_response.json()
            
            return templates.TemplateResponse(
                "admin.html",
                {
                    "request": request,
                    "current_boosts": current_boosts
                }
            )
        except Exception as e:
            print(f"Admin page error: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.post("/admin/boost")
async def add_boost(
    request: Request,
    article_id: int = Form(...),
    boost_factor: float = Form(...),
    start_time: str = Form(...),
    end_time: str = Form(...),
    boost_type: str = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Add the boost
            boost_response = await client.post(
                f"{API_URL}/articles/boosts",
                json={
                    "article_id": article_id,
                    "boost_factor": boost_factor,
                    "start_time": start_time,
                    "end_time": end_time,
                    "boost_type": boost_type
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if boost_response.status_code != 200:
                print(f"Boost error: {boost_response.text}")
                
        except Exception as e:
            print(f"Add boost error: {str(e)}")
            
    return RedirectResponse(url="/admin", status_code=303)

@app.post("/admin/boost/{article_id}/delete")
async def delete_boost(request: Request, article_id: int):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Delete the boost
            delete_response = await client.delete(
                f"{API_URL}/articles/boosts/{article_id}",
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if delete_response.status_code != 200:
                print(f"Delete boost error: {delete_response.text}")
                
        except Exception as e:
            print(f"Delete boost error: {str(e)}")
            
    return RedirectResponse(url="/admin", status_code=303)

@app.get("/add-article")
async def add_article_page(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    return templates.TemplateResponse(
        "add_article.html",
        {"request": request}
    )

@app.post("/add-article")
async def add_article(
    request: Request,
    title: str = Form(...),
    text: str = Form(...),
    authors: str = Form(...),
    tags: str = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info for author
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            user_data = user_response.json()
            
            # Process authors and tags
            authors_list = [author.strip() for author in authors.split(',')]
            if user_data['username'] not in authors_list:
                authors_list.append(user_data['username'])  # Add current user as author
            
            tags_list = [tag.strip() for tag in tags.split(',')]
            
            # Add article through API
            response = await client.post(
                f"{API_URL}/articles/",
                json={
                    "title": title,
                    "text": text,
                    "authors": authors_list,
                    "tags": tags_list,
                    "timestamp": datetime.now().isoformat()
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code == 200:
                return RedirectResponse(url="/dashboard", status_code=303)
            else:
                return templates.TemplateResponse(
                    "add_article.html",
                    {
                        "request": request,
                        "error": "Failed to add article"
                    }
                )
                
        except Exception as e:
            print(f"Add article error: {str(e)}")
            return templates.TemplateResponse(
                "add_article.html",
                {
                    "request": request,
                    "error": str(e)
                }
            )

@app.get("/edit-interests")
async def edit_interests_page(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            user_data = user_response.json()
            
            # Get user's current interests
            interests_response = await client.get(
                f"{API_URL}/users/{user_data['username']}/interests",
                headers={"Authorization": f"Bearer {token}"}
            )
            user_interests = []
            if interests_response.status_code == 200:
                user_interests = interests_response.json().get("interests", [])
            
            # Get all available tags
            tags_response = await client.get(
                f"{API_URL}/articles/popular-tags?n_tags=30",  # Get top 30 tags
                headers={"Authorization": f"Bearer {token}"}
            )
            available_tags = tags_response.json() if tags_response.status_code == 200 else []
            
            return templates.TemplateResponse(
                "edit_interests.html",
                {
                    "request": request,
                    "user": user_data,
                    "user_interests": user_interests,
                    "available_tags": available_tags
                }
            )
        except Exception as e:
            print(f"Edit interests error: {str(e)}")
            return RedirectResponse(url="/dashboard")

@app.post("/save-interests")
async def save_interests(
    request: Request,
    selected_tags: List[str] = Form(...)
):
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/")
        
    async with httpx.AsyncClient() as client:
        try:
            # Get user info
            user_response = await client.get(
                f"{API_URL}/users/me",
                headers={"Authorization": f"Bearer {token}"}
            )
            if user_response.status_code != 200:
                return RedirectResponse(url="/")
            
            user_data = user_response.json()
            
            # Save user interests
            interests_response = await client.post(
                f"{API_URL}/users/interests",
                json={
                    "username": user_data["username"],
                    "interests": selected_tags
                },
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if interests_response.status_code == 200:
                return RedirectResponse(url="/profile", status_code=303)
            else:
                return RedirectResponse(url="/edit-interests", status_code=303)
                
        except Exception as e:
            print(f"Save interests error: {str(e)}")
            return RedirectResponse(url="/edit-interests")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001) 