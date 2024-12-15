from google.cloud import firestore, storage
import requests
from google.cloud.firestore import FieldFilter as fil
import time
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import os

class DBHandle:
  def __init__(self, dbname, credentials_path='hchuong-firebase-adminsdk-1m82k-a70c60ad91.json') -> None:
    self.dbName = dbname

    # Nếu có file credentials được truyền vào, sử dụng nó
    import firebase_admin
    from firebase_admin import credentials, firestore, storage

    # Kiểm tra xem Firebase app đã được khởi tạo chưa
    try:
      firebase_admin.get_app()
    except ValueError:
      cred = credentials.Certificate(credentials_path)
      firebase_admin.initialize_app(cred)

    self.db = firestore.client()
    self.bucket = storage.bucket("hchuong.appspot.com")

  def insert(self, data: dict):
    try:
      new_ref = self.db.collection(self.dbName).add(data)
      new_reff = (new_ref[0]) 
      print('Time insert:', new_ref[0].ToDatatime())
      return new_ref[1].id
    except Exception as e:
      return False

  def update(self, id, data: dict):
    try:
      return self.db.collection(self.dbName).document(id).update(data)
    except Exception as e:
      return False

  def get_all(self):
    try:
      return self.db.collection(self.dbName).stream()
    except Exception as e:
      return False

  def get_by_id(self, id):
    try:
      return self.db.collection(self.dbName).document(id).get()
    except Exception as e:
      return False

  def delete(self, id):
    try:
      return self.db.collection(self.dbName).document(id).delete()
    except Exception as e:
      return False      
  
  def upload_file(self, file: UploadedFile, path: str):
    try:
      # Tạo tên file với đường dẫn đầy đủ
      file_path = f"{path}/{file.name}"
      
      # Upload file
      blob = self.bucket.blob(file_path)
      blob.upload_from_file(file, content_type=file.type)
      return True
    except Exception as e:
      print(f"Error uploading file: {str(e)}")
      return False